#!/bin/bash
# Generic environment normalizer. Load/export the site stack first, then source this.

if [ -n "${BASH_SOURCE:-}" ]; then
    _celephais_env_file="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _celephais_env_file="${(%):-%x}"
else
    _celephais_env_file="$0"
fi
_celephais_env_dir="$(cd "$(dirname "$_celephais_env_file")" && pwd)"
# shellcheck source=cmake/env/celephais_env_common.sh
source "${_celephais_env_dir}/celephais_env_common.sh"

celephais_module_init || true
celephais_add_default_runtime_paths

unset _celephais_env_dir _celephais_env_file
