docker-compose down
cd ../tmunan
docker build -t himmelroman/stream_app:latest -f stream_app/Dockerfile .
docker build -t himmelroman/imagine_app:latest -f imagine_app/Dockerfile .
cd .docker
docker-compose up
