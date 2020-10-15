import jetbrains.buildServer.configs.kotlin.v2019_2.*
import jetbrains.buildServer.configs.kotlin.v2019_2.buildFeatures.sshAgent
import jetbrains.buildServer.configs.kotlin.v2019_2.buildSteps.ScriptBuildStep
import jetbrains.buildServer.configs.kotlin.v2019_2.buildSteps.script
import jetbrains.buildServer.configs.kotlin.v2019_2.triggers.schedule
import jetbrains.buildServer.configs.kotlin.v2019_2.triggers.vcs
import jetbrains.buildServer.configs.kotlin.v2019_2.vcs.GitVcsRoot

/*
The settings script is an entry point for defining a TeamCity
project hierarchy. The script should contain a single call to the
project() function with a Project instance or an init function as
an argument.

VcsRoots, BuildTypes, Templates, and subprojects can be
registered inside the project using the vcsRoot(), buildType(),
template(), and subProject() methods respectively.

To debug settings scripts in command-line, run the

    mvnDebug org.jetbrains.teamcity:teamcity-configs-maven-plugin:generate

command and attach your debugger to the port 8000.

To debug in IntelliJ Idea, open the 'Maven Projects' tool window (View
-> Tool Windows -> Maven Projects), find the generate task node
(Plugins -> teamcity-configs -> teamcity-configs:generate), the
'Debug' option is available in the context menu for the task.
*/

version = "2020.1"

project {

    vcsRoot(Git)

    buildType(GithubActions)
    buildType(Test)
}

object GithubActions : BuildType({
    name = "GithubActions"

    vcs {
        root(Git)
    }

    steps {
        script {
            name = "TagBot"
            scriptContent = """
                set -x
                export GITHUB_REPOSITORY=`git config --get remote.origin.url`
                python -m tagbot.action
            """.trimIndent()
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
            dockerPull = true
            dockerImage = "docker.int.avast.com/skunk/base/tagbot:latest"
            param("org.jfrog.artifactory.selectedDeployableServer.downloadSpecSource", "Job configuration")
            param("org.jfrog.artifactory.selectedDeployableServer.useSpecs", "false")
            param("org.jfrog.artifactory.selectedDeployableServer.uploadSpecSource", "Job configuration")
        }
        script {
            name = "CompatHelper"
            scriptContent = """
                set -x
                
                export GITHUB_URL=`git config --get remote.origin.url`
                export BASE_URL="git@git.int.avast.com:"
                export GITHUB_REPOSITORY=${'$'}{GITHUB_URL#"${'$'}BASE_URL"}
                
                printenv
                
                julia /app/compat_helper/run_avast.jl
            """.trimIndent()
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
            dockerPull = true
            dockerImage = "docker.int.avast.com/skunk/base/compat_helper:latest"
            dockerRunParameters = """
                -v "${'$'}HOME/.ssh:/root/.ssh"
                -v "${'$'}SSH_AUTH_SOCK:/tmp/ssh_auth_sock"
                -e "SSH_AUTH_SOCK=/tmp/ssh_auth_sock"
            """.trimIndent()
            param("org.jfrog.artifactory.selectedDeployableServer.downloadSpecSource", "Job configuration")
            param("org.jfrog.artifactory.selectedDeployableServer.useSpecs", "false")
            param("org.jfrog.artifactory.selectedDeployableServer.uploadSpecSource", "Job configuration")
        }
    }

    triggers {
        vcs {
            enabled = false
        }
        schedule {
            schedulingPolicy = cron {
                dayOfWeek = "*"
            }
            branchFilter = ""
            triggerBuild = always()
            withPendingChangesOnly = false
        }
    }

    features {
        sshAgent {
            teamcitySshKey = "svc-aivision-cloner"
        }
    }

    requirements {
        startsWith("system.agent.name", "cloud-threatlabs-")
    }
})

object Test : BuildType({
    name = "Test"

    allowExternalStatus = true

    vcs {
        root(Git)
    }

    steps {
        script {
            name = "install packages and run tests"
            scriptContent = """
                set -x
                printenv
                julia --color=yes --project=@. -e 'using Pkg; pkg"instantiate"; pkg"test"'
            """.trimIndent()
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
            dockerImage = "docker.int.avast.com/skunk/base/julia/latest"
            dockerRunParameters = """
                -v "${'$'}HOME/.ssh:/root/.ssh"
                -v "${'$'}SSH_AUTH_SOCK:/tmp/ssh_auth_sock"
                -e "SSH_AUTH_SOCK=/tmp/ssh_auth_sock"
            """.trimIndent()
            param("org.jfrog.artifactory.selectedDeployableServer.downloadSpecSource", "Job configuration")
            param("org.jfrog.artifactory.selectedDeployableServer.useSpecs", "false")
            param("org.jfrog.artifactory.selectedDeployableServer.uploadSpecSource", "Job configuration")
        }
    }

    triggers {
        vcs {
        }
    }

    features {
        sshAgent {
            teamcitySshKey = "svc-aivision-cloner"
        }
    }

    requirements {
        startsWith("system.agent.name", "cloud-threatlabs-")
    }
})

object Git : GitVcsRoot({
    name = "git@git.int.avast.com:skunk/ExplainMill.jl.git"
    url = "git@git.int.avast.com:skunk/ExplainMill.jl.git"
    branchSpec = "+:*"
    authMethod = uploadedKey {
        userName = "git"
        uploadedKey = "svc-aivision-cloner"
    }
})
