docker-compose down
cd ../tmunan
docker build -t himmelroman/stream_app -f stream_app/Dockerfile .
docker build -t himmelroman/imagine_app -f imagine_app/Dockerfile .
cd .docker
docker-compose up
