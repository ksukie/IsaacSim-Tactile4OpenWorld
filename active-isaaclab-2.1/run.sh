#!/usr/bin/env bash

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces when the terminal supports it.
tabs 4 2>/dev/null || true

# get source directory
#export ISAACLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# get source directory of OpenWorldTactile
export OWT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#==
# Helper functions
#==

# check if running in docker
is_docker() {
    [ -f /.dockerenv ] || \
    grep -q docker /proc/1/cgroup || \
    [[ $(cat /proc/1/comm) == "containerd-shim" ]] || \
    grep -q docker /proc/mounts || \
    [[ "$(hostname)" == *"."* ]]
}

extract_isaacsim_path() {
    # Use the sym-link path to Isaac Sim directory
    local isaac_path=${ISAACLAB_PATH}/_isaac_sim
    # If above path is not available, try to find the path using python
    if [ ! -d "${isaac_path}" ]; then
        # Use the python executable to get the path
        local python_exe=$(extract_python_exe)
        # Retrieve the path importing isaac sim and getting the environment path
        if [ $(${python_exe} -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
            local isaac_path=$(${python_exe} -c "import isaacsim; import os; print(os.environ['ISAAC_PATH'])")
        fi
    fi
    # check if there is a path available
    if [ ! -d "${isaac_path}" ]; then
        # throw an error if no path is found
        echo -e "[ERROR] Unable to find the Isaac Sim directory: '${isaac_path}'" >&2
        echo -e "\tThis could be due to the following reasons:" >&2
        echo -e "\t1. Conda environment is not activated." >&2
        echo -e "\t2. Isaac Sim pip package 'isaacsim-rl' is not installed." >&2
        echo -e "\t3. Isaac Sim directory is not available at the default path: ${ISAACLAB_PATH}/_isaac_sim" >&2
        # exit the script
        exit 1
    fi
    # return the result
    echo ${isaac_path}
}

# extract the python from isaacsim
extract_python_exe() {
    # check if using conda
    if ! [[ -z "${CONDA_PREFIX}" ]]; then
        # use conda python
        local python_exe=${CONDA_PREFIX}/bin/python
    else
        # use kit python
        local python_exe=${ISAACLAB_PATH}/_isaac_sim/python.sh

    if [ ! -f "${python_exe}" ]; then
            # note: we need to check system python for cases such as docker
            # inside docker, if user installed into system python, we need to use that
            # otherwise, use the python from the kit
            if [ $(python -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
                local python_exe=$(which python)
            fi
        fi
    fi
    # check if there is a python path available
    if [ ! -f "${python_exe}" ]; then
        echo -e "[ERROR] Unable to find any Python executable at path: '${python_exe}'" >&2
        echo -e "\tThis could be due to the following reasons:" >&2
        echo -e "\t1. Conda environment is not activated." >&2
        echo -e "\t2. Isaac Sim pip package 'isaacsim-rl' is not installed." >&2
        echo -e "\t3. Python executable is not available at the default path: ${ISAACLAB_PATH}/_isaac_sim/python.sh" >&2
        exit 1
    fi
    # return the result
    echo ${python_exe}
}

# extract the simulator exe from isaacsim
extract_isaacsim_exe() {
    # obtain the isaac sim path
    local isaac_path=$(extract_isaacsim_path)
    # isaac sim executable to use
    local isaacsim_exe=${isaac_path}/isaac-sim.sh
    # check if there is a python path available
    if [ ! -f "${isaacsim_exe}" ]; then
        # check for installation using Isaac Sim pip
        # note: pip installed Isaac Sim can only come from a direct
        # python environment, so we can directly use 'python' here
        if [ $(python -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
            # Isaac Sim - Python packages entry point
            local isaacsim_exe="isaacsim isaacsim.exp.full"
        else
            echo "[ERROR] No Isaac Sim executable found at path: ${isaac_path}" >&2
            exit 1
        fi
    fi
    # return the result
    echo ${isaacsim_exe}
}

# check if input directory is a python extension and install the module
install_isaaclab_extension() {
    # retrieve the python executable
    python_exe=$(extract_python_exe)
    # if the directory contains setup.py then install the python module
    if [ -f "$1/setup.py" ]; then
        echo -e "\t module: $1"
        ${python_exe} -m pip install --editable $1 -v
    fi
}

has_python_arg() {
    local arg_name=$1
    shift
    for arg in "$@"; do
        if [[ "${arg}" == "${arg_name}" || "${arg}" == "${arg_name}="* ]]; then
            return 0
        fi
    done
    return 1
}

python_uses_app_launcher_args() {
    local script_path=$1
    if [[ -z "${script_path}" || "${script_path}" == -* ]]; then
        return 1
    fi
    if [ ! -f "${script_path}" ] && [ -f "${OWT_PATH}/${script_path}" ]; then
        script_path="${OWT_PATH}/${script_path}"
    fi
    if [ -f "${script_path}" ] && grep -q "AppLauncher.add_app_launcher_args" "${script_path}"; then
        return 0
    fi
    return 1
}

# setup anaconda environment for Isaac Lab
setup_conda_env() {
    # get environment name from input
    local env_name=$1
    # check conda is installed
    if ! command -v conda &> /dev/null
    then
        echo "[ERROR] Conda could not be found. Please install conda and try again."
        exit 1
    fi

    # check if the environment exists
    if { conda env list | grep -w ${env_name}; } >/dev/null 2>&1; then
        echo -e "[INFO] Conda environment named '${env_name}' already exists."
    else
        echo -e "[INFO] Creating conda environment named '${env_name}'..."
        conda create -y --name ${env_name} python=3.10
    fi

    # cache current paths for later
    cache_pythonpath=$PYTHONPATH
    cache_ld_library_path=$LD_LIBRARY_PATH
    # clear any existing files
    rm -f ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    rm -f ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh
    # activate the environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${env_name}
    # setup directories to load Isaac Sim variables
    mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
    mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d

    # add variables to environment during activation
    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# for Isaac Lab' \
        'export ISAACLAB_PATH='${ISAACLAB_PATH}'' \
        'alias isaaclab='${ISAACLAB_PATH}'/isaaclab.sh' \
        '' \
        '# show icon if not running headless' \
        'export RESOURCE_NAME="IsaacSim"' \
        '' > ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh

    # check if we have _isaac_sim directory -> if so that means binaries were installed.
    # we need to setup conda variables to load the binaries
    local isaacsim_setup_conda_env_script=${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh

    if [ -f "${isaacsim_setup_conda_env_script}" ]; then
        # add variables to environment during activation
        printf '%s\n' \
            '# for Isaac Sim' \
            'source '${isaacsim_setup_conda_env_script}'' \
            '' >> ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    fi

    # reactivate the environment to load the variables
    # needed because deactivate complains about Isaac Lab alias since it otherwise doesn't exist
    conda activate ${env_name}

    # remove variables from environment during deactivation
    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# for Isaac Lab' \
        'unalias isaaclab &>/dev/null' \
        'unset ISAACLAB_PATH' \
        '' \
        '# restore paths' \
        'export PYTHONPATH='${cache_pythonpath}'' \
        'export LD_LIBRARY_PATH='${cache_ld_library_path}'' \
        '' \
        '# for Isaac Sim' \
        'unset RESOURCE_NAME' \
        '' > ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh

    # check if we have _isaac_sim directory -> if so that means binaries were installed.
    if [ -f "${isaacsim_setup_conda_env_script}" ]; then
        # add variables to environment during activation
        printf '%s\n' \
            '# for Isaac Sim' \
            'unset CARB_APP_PATH' \
            'unset EXP_PATH' \
            'unset ISAAC_PATH' \
            '' >> ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh
    fi

    # install some extra dependencies
    echo -e "[INFO] Installing extra dependencies (this might take a few minutes)..."
    conda install -c conda-forge -y importlib_metadata &> /dev/null

    # deactivate the environment
    conda deactivate
    # add information to the user about alias
    echo -e "[INFO] Added 'isaaclab' alias to conda environment for 'isaaclab.sh' script."
    echo -e "[INFO] Created conda environment named '${env_name}'.\n"
    echo -e "\t\t1. To activate the environment, run:                conda activate ${env_name}"
    echo -e "\t\t2. To install Isaac Lab extensions, run:            isaaclab -i"
    echo -e "\t\t4. To perform formatting, run:                      isaaclab -f"
    echo -e "\t\t5. To deactivate the environment, run:              conda deactivate"
    echo -e "\n"
}

# update the vscode settings from template and isaac sim settings
update_vscode_settings() {
    echo "[INFO] Setting up vscode settings..."
    # retrieve the python executable
    python_exe=$(extract_python_exe)
    # path to setup_vscode.py
    setup_vscode_script="${ISAACLAB_PATH}/.vscode/tools/setup_vscode.py"
    # check if the file exists before attempting to run it
    if [ -f "${setup_vscode_script}" ]; then
        ${python_exe} "${setup_vscode_script}"
    else
        echo "[WARNING] Unable to find the script 'setup_vscode.py'. Aborting vscode settings setup."
    fi
}

# print the usage description
print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [-i] [-f] [-p] [-s] [-t] [-v] [-c] -- Utility to manage Isaac Lab."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help             Display the help content."
    echo -e "\t-i, --install [all]    Install the OpenWorldTactile core packages. Use 'all' to also install all extra packages [openworldtactile_uipc]."
    echo -e "\t-f, --format           Run pre-commit to format the code and check lints."
    echo -e "\t-p, --python           Run the python executable provided by Isaac Sim or virtual environment (if active)."
    echo -e "\t-s, --sim              Run the simulator executable (isaac-sim.sh) provided by Isaac Sim."
    echo -e "\t-t, --test             Run the project's Python test suites."
    echo -e "\t-v, --vscode           Generate the VSCode settings file from template."
    echo -e "\t-c, --conda [NAME]     Create the Isaac Lab conda environment (default: env_isaaclab)."
    echo -e "\n" >&2
}


#==
# Main
#==

# check argument provided
if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    print_help
    exit 1
fi

# pass the arguments
while [[ $# -gt 0 ]]; do
    # read the key
    case "$1" in
        -i|--install)
            # install the python packages in openworldtactile/packages directory
            echo "[INFO] Installing extensions inside the OpenWorldTactile repository..."
            python_exe=$(extract_python_exe)
            # recursively look into directories and install them
            # this does not check dependencies between extensions
            export -f extract_python_exe
            export -f install_isaaclab_extension
            # source directory
            # find -L "${OWT_PATH}/packages" -mindepth 1 -maxdepth 1 -type d -exec bash -c 'install_isaaclab_extension "{}"' \;
            # install core packages
            echo "[INFO] Installing package [openworldtactile]..."
            ${python_exe} -m pip install -e ${OWT_PATH}/packages/core

            echo "[INFO] Installing package [openworldtactile_assets]..."
            ${python_exe} -m pip install -e ${OWT_PATH}/packages/assets

            echo "[INFO] Installing package [openworldtactile_tasks]..."
            ${python_exe} -m pip install -e ${OWT_PATH}/packages/tasks

            if [ -z "$2" ]; then
                echo "[INFO] No extra packages installed."
            elif [ "$2" = "all" ]; then
                echo "[INFO] Installing package openworldtactile_uipc..."
                ${python_exe} -m pip install -e ${OWT_PATH}/packages/uipc -v
                # consume the extra argument so it isn't processed by the outer loop
                shift
            # else
            #     echo "[INFO] Installing rl-framework: $2"
            #     extension_name=$2
            #     shift # past argument
            fi

            # check if we are inside a docker container or are building a docker image
            # in that case don't setup VSCode since it asks for EULA agreement which triggers user interaction
            # if is_docker; then
            #     echo "[INFO] Running inside a docker container. Skipping VSCode settings setup."
            #     echo "[INFO] To setup VSCode settings, run 'isaaclab -v'."
            # else
            #     # update the vscode settings
            #     update_vscode_settings
            # fi

            # unset local variables
            unset extract_python_exe
            unset install_isaaclab_extension
            shift # past argument
            ;;
        -c|--conda)
            # use default name if not provided
            if [ -z "$2" ]; then
                echo "[INFO] Using default conda environment name: env_isaaclab"
                conda_env_name="env_isaaclab"
            else
                echo "[INFO] Using conda environment name: $2"
                conda_env_name=$2
                shift # past argument
            fi
            # setup the conda environment for Isaac Lab
            setup_conda_env ${conda_env_name}
            shift # past argument
            ;;
        -f|--format)
            # reset the python path to avoid conflicts with pre-commit
            # this is needed because the pre-commit hooks are installed in a separate virtual environment
            # and it uses the system python to run the hooks
            if [ -n "${CONDA_DEFAULT_ENV}" ]; then
                cache_pythonpath=${PYTHONPATH}
                export PYTHONPATH=""
            fi
            # run the formatter over the repository
            # check if pre-commit is installed
            if ! command -v pre-commit &>/dev/null; then
                echo "[INFO] Installing pre-commit..."
                pip install pre-commit
            fi
            # always execute inside the OpenWorldTactile directory
            echo "[INFO] Formatting the repository..."
            cd ${OWT_PATH}
            pre-commit run --all-files
            cd - > /dev/null
            # set the python path back to the original value
            if [ -n "${CONDA_DEFAULT_ENV}" ]; then
                export PYTHONPATH=${cache_pythonpath}
            fi
            shift # past argument
            # exit neatly
            break
            ;;
        -p|--python)
            # run the python provided by isaacsim
            python_exe=$(extract_python_exe)
            echo "[INFO] Using python from: ${python_exe}"
            shift # past argument
            python_args=("$@")
            if python_uses_app_launcher_args "${python_args[0]}" && ! has_python_arg "--rendering_mode" "${python_args[@]}"; then
                echo "[INFO] Defaulting --rendering_mode to performance."
                python_args+=("--rendering_mode" "performance")
            fi
            ${python_exe} "${python_args[@]}"
            # exit neatly
            break
            ;;
        -s|--sim)
            # run the simulator exe provided by isaacsim
            isaacsim_exe=$(extract_isaacsim_exe)
            echo "[INFO] Running isaac-sim from: ${isaacsim_exe}"
            shift # past argument
            ${isaacsim_exe} --ext-folder ${OWT_PATH}/packages $@
            # exit neatly
            break
            ;;
        -t|--test)
            # run the python provided by isaacsim
            python_exe=$(extract_python_exe)
            shift # past argument
            ${python_exe} -m pytest \
                ${OWT_PATH}/packages/core/test \
                ${OWT_PATH}/packages/tasks/test \
                ${OWT_PATH}/experiments/tactile-bench $@
            # exit neatly
            break
            ;;
        -v|--vscode)
            # update the vscode settings
            update_vscode_settings
            shift # past argument
            # exit neatly
            break
            ;;
        -h|--help)
            print_help
            exit 1
            ;;
        *) # unknown option
            echo "[Error] Invalid argument provided: $1"
            print_help
            exit 1
            ;;
    esac
done
