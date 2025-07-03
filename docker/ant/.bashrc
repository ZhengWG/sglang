# .bashrc

# User specific aliases and functions

alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi
export PS1='\n\e[1;37m[\e[m\e[1;32m\u\e[m\e[1;33m@\e[m\e[1;35m\H\e[m \e[4m`pwd`\e[m\e[1;37m]\e[m\e[1;36m\e[m\n\$'
export EDITOR=vim
export PATH=$PATH:/opt/satools
alias vi='vim'

umask 002
export PATH=/opt/conda/bin:$PATH



export LANG=zh_CN.UTF-8
export LANGUAGE=zh_cn
export LESSCHARSET=utf-8
export LC_ALL=zh_CN.UTF-8
export LC_CTYPE=zh_CN.UTF-8
export LC_NUMERIC=zh_CN.UTF-8
export LC_TIME=zh_CN.UTF-8
export LC_COLLATE=zh_CN.UTF-8
export LC_MONETARY=zh_CN.UTF-8
export LC_MESSAGES=zh_CN.UTF-8
export LC_PAPER=zh_CN.UTF-8
export LC_NAME=zh_CN.UTF-8
export LC_ADDRESS=zh_CN.UTF-8
export LC_TELEPHONE=zh_CN.UTF-8
export LC_MEASUREMENT=zh_CN.UTF-8
export LC_IDENTIFICATION=zh_CN.UTF-8

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64


export MC_CUSTOM_TOPO_JSON=/etc/transfer-engine/h20-4nic-topo.json
source /root/.xccl_env.sh

