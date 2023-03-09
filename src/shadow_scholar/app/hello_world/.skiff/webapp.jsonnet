/**
 * This file defines the infrastructure we need to run the app on Kubernetes.
 *
 * For more information on the JSONNET language, see:
 * https://jsonnet.org/learning/getting_started.html
 */

local common = import '../../common/common.jsonnet';

function(image, cause, sha, env, branch, repo, buildId)
    // This tells Kubernetes what resources we need to run.
    // For more information see:
    // https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/\#resource-units-in-kubernetes
    local cpu = '50m';
    local memory = '200Mi';
    local gpu = ''; // 'k80', 'p100', 't4x4', etc.
    local environmentVariables = [
    //    { name: "NAME", value: "VALUE" },
    //    {
    //        name: "MY_SECRET",
    //        valueFrom: {
    //            secretKeyRef: {
    //                name: "k8s-secret-name",
    //                key: "password"
    //            }
    //        }
    //    },
    ];
    common.ShadowApp('hello-world', image, cause, sha, cpu, memory, env, branch, repo, buildId, gpu, environmentVariables)
    
