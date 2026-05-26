#!/bin/sh

NV_DRIVER_VERS=$(sed -n 's/^NVRM.*Kernel Module\( for [a-z0-9_]*\| \) *\([^() ]*\).*$/\2/p' /proc/driver/nvidia/version 2>/dev/null | sed 's/^$/unknown/')
_CUDA_COMPAT_CHECKFILE="${HOME}/.${NV_DRIVER_VERS}.$(hostname).checked"

if [ -f "${_CUDA_COMPAT_CHECKFILE}" ]; then
  . "${_CUDA_COMPAT_CHECKFILE}"
  return 0 2>/dev/null || exit 0
fi

_RUNNING_CUDA_VERSION="$(nvidia-smi -q -d COMPUTE 2>/dev/null | grep "^CUDA Version" | sed 's/^.*: //')"
_CONTAINER_CUDA_VERSION="$(echo "${CUDA_VERSION}" | cut -d . -f 1-2)"

if [ -e /dev/nvgpu ] && [ "${_RUNNING_CUDA_VERSION:-}" = "${_CONTAINER_CUDA_VERSION:-}" ]; then
  # skip compat check to WAR http://nvbugs/4472547
  touch "${_CUDA_COMPAT_CHECKFILE}" 2>/dev/null
fi

# If the CUDA driver was detected and the compat check hasn't been flagged as done yet, proceed
if { { [ -n "${NV_DRIVER_VERS}" ] && [ -e /dev/nvidiactl ] } || [ -e /dev/nvgpu ] } && [ ! -e "${_CUDA_COMPAT_CHECKFILE}" ]; then

  # find cuda_compat with highest version or with CUDA_VERSION
  if [ -z "${_CUDA_COMPAT_PATH:-}" ]; then
    _CUDA_HOMES=$(find /usr/local/ -maxdepth 1 -type d -name "cuda-${_CONTAINER_CUDA_VERSION}*" | sort -r)
    for cuda_home in ${_CUDA_HOMES:-}; do
      _libcuda_compat_path=$(find "${cuda_home}/compat" -type f -name "libcuda.so.*" 2>/dev/null | head -n 1)
      if [ -n "${_libcuda_compat_path}" ]; then
        export _CUDA_COMPAT_PATH=$(dirname "${_libcuda_compat_path}")
        break
      fi
    done
  fi

  if [ -n "${_CUDA_COMPAT_PATH:-}" ]; then
    # Make note that compat check has already run in this environment
    cat <<EOF > "${_CUDA_COMPAT_CHECKFILE}"
export LD_LIBRARY_PATH="${_CUDA_COMPAT_PATH}\${LD_LIBRARY_PATH:+":\${LD_LIBRARY_PATH}"}"
EOF
  fi
fi

if [ -s "${_CUDA_COMPAT_CHECKFILE}" ]; then
  . "${_CUDA_COMPAT_CHECKFILE}"
fi

# Clean up
unset _CUDA_HOMES
unset _CUDA_COMPAT_CHECKFILE
unset _RUNNING_CUDA_VERSION
unset _CONTAINER_CUDA_VERSION
unset _libcuda_compat_path
