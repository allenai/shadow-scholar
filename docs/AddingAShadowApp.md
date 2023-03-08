# Adding a Shadow App

This document outlines how to deploy a new Shadow app to Skiff.

After you've added your app to the `shadow-scholar` repository under `src/shadow_scholar/app/`,
you'll need to add a Skiff configuration for it.

```bash
$ export APP_NAME=hello-world
$ export APP_DIR=hello_world
$ mkdir src/shadow_scholar/app/$APP_DIR/.skiff
$ cat <<EOF > src/shadow_scholar/app/$APP_DIR/.skiff/webapp.jsonnet
/**
 * This file defines the infrastructure we need to run the app on Kubernetes.
 *
 * For more information on the JSONNET language, see:
 * https://jsonnet.org/learning/getting_started.html
 */

local common = import '../../common/.skiff/common.jsonnet';

function(image, cause, sha, env, branch, repo, buildId)
    // This tells Kubernetes what resources we need to run.
    // For more information see:
    // https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#resource-units-in-kubernetes
    local cpu = '50m';
    local memory = '200Mi';
    common.ShadowApp('$APP_NAME', image, cause, sha, cpu, memory, env, branch, repo, buildId)
    
EOF
```

Then you'll need to add a Cloud Build trigger for the app (note this will only build when the included files are changed).
This step requires permissions to create triggers in the `ai2-reviz` project, so you may have to ask ReViz to do it.

```bash
gcloud beta builds triggers create github --project=ai2-reviz --trigger-config <(cat <<EOF
{
    "name": "deploy-shadow-scholar-$APP_NAME-prod",
    "description": "Deploy $APP_NAME to production",
    "github": {
        "owner": "allenai",
        "name": "shadow-scholar",
        "push": {
          "branch": "^main$"
        }
    },
    "substitutions": {
        "_ENV": "prod",
        "_SUBMODULE_PATH": "$APP_DIR"
    },
    "includedFiles": [
        "app/**",
        "src/shadow_scholar/app/common/**",
        "src/shadow_scholar/app/$APP_DIR/**",
        "src/shadow_scholar/collections/**",
        "src/shadow_scholar/__init__.py",
        "src/shadow_scholar/__main__.py",
        "src/shadow_scholar/cli.py",
        "pyproject.toml",
        "skiff.json"
    ],
    "filename": "src/shadow_scholar/app/common/cloudbuild-deploy.yaml"
}
EOF
)
```

After this, changes to the above files on `main` will trigger new builds for the new app.
