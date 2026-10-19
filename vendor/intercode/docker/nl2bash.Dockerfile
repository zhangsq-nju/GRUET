FROM ubuntu:latest

ARG file_system_version=1

RUN apt-get update && \
    apt-get install -y bash python3 psmisc bsdmainutils cron imagemagick dnsutils git tree net-tools iputils-ping coreutils curl cpio jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV file_system_version=${file_system_version}
COPY docker/bash_scripts/setup_nl2b_fs_*.sh /
RUN chmod +x /setup_nl2b_fs_${file_system_version}.sh
RUN /setup_nl2b_fs_${file_system_version}.sh

COPY docker/docker.gitignore /
RUN mv docker.gitignore .gitignore
RUN git config --global user.email "intercode@pnlp.org"
RUN git config --global user.name "intercode"
RUN git init
RUN git add -A
RUN git commit -m 'initial commit'

WORKDIR /
