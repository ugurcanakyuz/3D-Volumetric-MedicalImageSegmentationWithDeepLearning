:: Start docker container to access Python virtual environment from PyCharm on another computer in th LAN.
docker-compose -p torchremote -f DockerFiles\compose-remote.yml up --build --remove-orphans
