docker-compose down
cd ../tmunan
git pull
docker build -t himmelroman/tmunan_stream:latest -f stream_app/Dockerfile .
docker build -t himmelroman/tmunan_image:latest -f imagine_app/Dockerfile .
cd ../.docker
docker-compose up
