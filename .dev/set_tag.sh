#!/usr/bin/env bash

if ! [ $# -eq 1 ]; then
    echo "Usage: $0 tag_name"
    exit 1
fi

tag_name=$1

commit_before=$(git rev-list -n 1 ${tag_name})
echo "$(date '+%F %T') Before changes, tag ${tag_name} is pointing to commit ${commit_before}" | tee -a ${HOME}/.git_tags_history

echo "Deleting tag: ${tag_name} from local branch..."
if git tag -d ${tag_name}; then
    echo "Deleting tag: ${tag_name} from remote branch..."
    git push origin :refs/tags/${tag_name}
fi

echo "Setting tag: ${tag_name} for local branch..."
if git tag -a ${tag_name} -m "${tag_name}" $(git rev-parse HEAD); then

    echo "Pushing tag: ${tag_name} for remote branch..."
    if git push origin ${tag_name}; then
        commit_after=$(git rev-list -n 1 ${tag_name})
        echo "$(date '+%F %T') After changes, tag ${tag_name} is pointing to commit ${commit_after}" | tee -a ${HOME}/.git_tags_history
        echo "Success!"
    fi
fi
