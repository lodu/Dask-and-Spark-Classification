#!/bin/bash
if [ "$1" == "start" ]; then
docker run -d -p 27017-27019:27017-27019 -e MONGO_INITDB_ROOT_USERNAME={{USERNAME}} -e MONGO_INITDB_ROOT_PASSWORD={{PASSWORD}} -v /home/{{USER}}/mongodb-docker:/data/db --rm --name mongodb mongo
fi

if [ "$1" == "stop" ]; then
        docker stop mongodb
fi
