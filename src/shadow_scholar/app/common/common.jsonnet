/**
 * This file contains common code that's used to generate the Kubernetes config for individual
 * model endpoints.
 *
 * For more information on the JSONNET language, see:
 * https://jsonnet.org/learning/getting_started.html
 */

local config = import '../../../../skiff.json';

{
    /**
     * @param id            {string}    A unique identifier for the application. This identifier
     *                                  must be URL safe, as it's used to determine what requests
     *                                  are routed to the endpoint. For instance, if the id is
     *                                  `foo`, requests beginning with `foo.shadow-scholar` are routed
     *                                  to the endpoint.
     * @param image         {string}    The image tag to deploy.
     * @param cause         {string}    A message describing the reason for the deployment.
     * @param sha           {string}    The git sha.
     * @param env           {string}    A unique identifier for the environment. This determines the
     *                                  the hostname that'll be used, the number of replicas, and more.
     *                                  If set the 'prod' this deploys to demo.allennlp.org.
     * @param branch        {string}    The branch name.
     * @param repo          {string}    The repo name.
     * @param buildId       {string}    The Google Cloud Build ID.
     */
    ShadowApp(id, image, cause, sha, cpu, memory, env, branch, repo, buildId):
        // A list of hostnames served by your application. By default your application's
        // `prod` environment will receive requests made to `$appName.apps.allenai.org` and
        // non-production environments will receive requests made to `$appName-$env.apps.allenai.org`.
        //
        // If you'd like to use a custom domain make sure DNS is pointed to the cluster's IP
        // address and add the domain to the `customDomains` list in `../skiff.json`.
        local hosts = util.getHosts(env, id, config, '.apps.allenai.org');

        // All Skiff applications get a *.allen.ai URL in addition to *.apps.allenai.org.
        // This domain is attached to a separate Ingress, as to support authentication
        // via either canonical domain.
        local allenAIHosts = util.getHosts(env, id, config, '.allen.ai');

        // In production you run should run two or more replicas of your
        // application, so that if one instance goes down or is busy (e.g., during
        // a deployment), users can still use the remaining replicas of your
        // application.
        //
        // However, if you use GPUs, which are expensive, consider setting the prod
        // replica count to 1 as a trade-off between availability and costs.
        //
        // In all other environments (e.g., adhocs) we run a single instance to
        // save money.
        local numReplicas = if env == 'prod' then config.replicas.prod else 1;

        // Each app gets it's own namespace.
        local namespaceName = config.appName;

        // Since we deploy resources for different environments in the same namespace,
        // we need to give things a fully qualified name that includes the environment
        // as to avoid unintentional collission / redefinition.
        local fullyQualifiedName = config.appName + '-' + id + '-' + env;

        // Every resource is tagged with the same set of labels. These labels serve the
        // following purposes:
        //  - They make it easier to query the resources, i.e.
        //      kubectl get pod -l app=my-app,env=staging
        //  - The service definition uses them to find the pods it directs traffic to.
        local namespaceLabels = {
            app: config.appName,
            contact: config.contact,
            team: config.team
        };

        local labels = namespaceLabels + {
            env:   env,
            appId: id
        };

        local selectorLabels = {
            app:   config.appName,
            env:   env,
            appId: id
        };

        // By default multiple instances of your application could get scheduled
        // to the same node. This means if that node goes down your application
        // does too. We use the label below to avoid that.
        local antiAffinityLabels = {
            onlyOneOfPerNode: fullyQualifiedName
        };
        local podLabels = labels + antiAffinityLabels;

        // Annotations carry additional information about your deployment that
        // we use for auditing, debugging and administrative purposes
        local annotations = {
            "apps.allenai.org/sha": sha,
            "apps.allenai.org/branch": branch,
            "apps.allenai.org/repo": repo,
            "apps.allenai.org/build": buildId
        };

        // Running on a GPU requires a special limit on the container, and a
        // specific nodeSelector.
        local gpuInConfig = std.count(std.objectFields(config), "gpu") > 0;

        // determine number of gpus
        local gpuLimits = if gpuInConfig then
            if config.gpu == "k80x2" then
                { 'nvidia.com/gpu': 2 }
            else if config.gpu == "t4x4" then
                { 'nvidia.com/gpu': 4 }
            else
                { 'nvidia.com/gpu': 1 }
        else {};

        local nodeSelector = if gpuInConfig then
            if config.gpu == "k80" || config.gpu == "k80x2" then
                { 'cloud.google.com/gke-accelerator': 'nvidia-tesla-k80' }
            else if config.gpu == "p100" then
                { 'cloud.google.com/gke-accelerator': 'nvidia-tesla-p100' }
            else if config.gpu == "t4x4" then
                { 'cloud.google.com/gke-accelerator': 'nvidia-tesla-t4' }
            else
                error "invalid GPU specification; expected 'k80', 'k80x2', 'p100' or 't4x4', but got: " + config.gpu
        else
             { };

        // The port the app is bound to.
        local appPort = 8000;
        // This is used to verify that the app is funtional.
        local appHealthCheck = {
            port: appPort,
            scheme: 'HTTP'
        };

        local namespace = {
            apiVersion: 'v1',
            kind: 'Namespace',
            metadata: {
                name: namespaceName,
                labels: namespaceLabels
            }
        };

        local tls = util.getTLSConfig(fullyQualifiedName, hosts);
        local ingress = {
            apiVersion: 'networking.k8s.io/v1',
            kind: 'Ingress',
            metadata: {
                name: fullyQualifiedName,
                namespace: namespaceName,
                labels: labels,
                annotations: annotations + tls.ingressAnnotations + util.getAuthAnnotations(config, '.apps.allenai.org') + {
                    'nginx.ingress.kubernetes.io/ssl-redirect': 'true'
                }
            },
            spec: {
                tls: [ tls.spec + { hosts: hosts } ],
                rules: [
                    {
                        host: host,
                        http: {
                            paths: [
                                {
                                    path: '/',
                                    pathType: 'Prefix',
                                    backend: {
                                        service: {
                                            name: fullyQualifiedName,
                                            port: {
                                                number: appPort
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    } for host in hosts
                ]
            }
        };

        local allenAIIngress = {
            apiVersion: 'networking.k8s.io/v1',
            kind: 'Ingress',
            metadata: {
                name: fullyQualifiedName + '-allen-dot-ai',
                namespace: namespaceName,
                labels: labels,
                annotations: annotations + util.getAuthAnnotations(config, '.allen.ai') + {
                    'nginx.ingress.kubernetes.io/ssl-redirect': 'true'
                }
            },
            spec: {
                tls: [ { hosts:  allenAIHosts} ],
                rules: [
                    {
                        host: host,
                        http: {
                            paths: [
                                {
                                    path: '/',
                                    pathType: 'Prefix',
                                    backend: {
                                        service: {
                                            name: fullyQualifiedName,
                                            port: {
                                                number: appPort
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    } for host in allenAIHosts
                ]
            }
        };

        local deployment = {
            apiVersion: 'apps/v1',
            kind: 'Deployment',
            metadata: {
                labels: labels,
                name: fullyQualifiedName,
                namespace: namespaceName,
                annotations: annotations + {
                    'kubernetes.io/change-cause': cause
                }
            },
            spec: {
                strategy: {
                    type: 'RollingUpdate',
                    rollingUpdate: {
                        maxSurge: numReplicas // This makes deployments faster.
                    }
                },
                revisionHistoryLimit: 3,
                replicas: numReplicas,
                selector: {
                    matchLabels: selectorLabels
                },
                template: {
                    metadata: {
                        name: fullyQualifiedName,
                        namespace: namespaceName,
                        labels: podLabels,
                        annotations: annotations
                    },
                    spec: {
                        # This block tells the cluster that we'd like to make sure
                        # each instance of your application is on a different node. This
                        # way if a node goes down, your application doesn't:
                        # See: https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#node-isolation-restriction
                        affinity: {
                            podAntiAffinity: {
                                requiredDuringSchedulingIgnoredDuringExecution: [
                                    {
                                       labelSelector: {
                                            matchExpressions: [
                                                {
                                                        key: labelName,
                                                        operator: "In",
                                                        values: [ antiAffinityLabels[labelName], ],
                                                } for labelName in std.objectFields(antiAffinityLabels)
                                           ],
                                        },
                                        topologyKey: "kubernetes.io/hostname"
                                    },
                                ]
                            },
                        },
                        nodeSelector: nodeSelector,
                        containers: [
                            {
                                name: fullyQualifiedName,
                                image: image,
                                # The "probes" below allow Kubernetes to determine
                                # if your application is working properly.
                                #
                                # The readinessProbe is used to determine if
                                # an instance of your application can accept live
                                # requests. The configuration below tells Kubernetes
                                # to stop sending live requests to your application
                                # if it returns 3 non 2XX responses over 30 seconds.
                                # When this happens the application instance will
                                # be taken out of rotation and given time to "catch-up".
                                # Once it returns a single 2XX, Kubernetes will put
                                # it back in rotation.
                                #
                                # Kubernetes also has a livenessProbe that can be used to restart
                                # deadlocked processes. You can find out more about it here:
                                # https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/#define-a-liveness-command
                                #
                                # We don't use a livenessProbe as it's easy to cause unnecessary
                                # restarts, which can be really disruptive to a site's availability.
                                # If you think your application is likely to be unstable after running
                                # for long periods send a note to reviz@allenai.org so we can work
                                # with you to craft the right livenessProbe.
                                readinessProbe: {
                                    httpGet: appHealthCheck + {
                                        path: '/?check=rdy'
                                    },
                                    periodSeconds: 10,
                                    failureThreshold: 3
                                },
                                # This tells Kubernetes what CPU and memory resources your API needs.
                                # We set these values low by default, as most applications receive
                                # bursts of activity and accordingly don't need dedicated resources
                                # at all times.
                                #
                                # Your application will be allowed to use more resources than what's
                                # specified below. But your application might be killed if it uses
                                # more than what's requested. If you know you need more memory
                                # or that your workload is CPU intensive, consider increasing the
                                # values below.
                                #
                                # For more information about these values, and the current maximums
                                # that your application can request, see:
                                # https://skiff.allenai.org/resources.html
                                resources: {
                                    requests: {
                                        cpu: cpu,
                                        memory: memory
                                    },
                                    limits: { }
                                       + gpuLimits # only the first container should have gpuLimits applied
                                }
                            }
                        ]
                    }
                }
            }
        };

        local service = {
            apiVersion: 'v1',
            kind: 'Service',
            metadata: {
                name: fullyQualifiedName,
                namespace: namespaceName,
                labels: labels,
                annotations: annotations
            },
            spec: {
                selector: selectorLabels,
                ports: [
                    {
                        port: appPort,
                        name: 'http'
                    }
                ]
            }
        };

        local pdb = {
            apiVersion: 'policy/v1beta1',
            kind: 'PodDisruptionBudget',
            metadata: {
                name: fullyQualifiedName,
                namespace: namespaceName,
                labels: labels,
            },
            spec: {
                minAvailable: if numReplicas > 1 then 1 else 0,
                selector: {
                    matchLabels: selectorLabels,
                },
            },
        };

        [
            namespace,
            ingress,
            allenAIIngress,
            deployment,
            service,
            pdb
        ]
}
