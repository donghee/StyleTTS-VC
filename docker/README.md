## Build Docker Image for styletts-vc 

Requirements: 
 - Ubuntu 20.04 or higher
 - NVIDIA driver 535 or higher
 - NVIDIA Container Toolkit aka nvidia-docker2

```
apt-get install nvidia-docker2
cd StyleTTS-VC/docker
./build.sh
```

## Run jupyter notebook

```
cd StyleTTS-VC/docker
./run.sh
cd Demo
jupyter-notebook  --allow-root --ip='*'
```

http://localhost:8888

---

## Create docker network

```
docker network create -d bridge aiavatar-network
```

## Connect docker network to segdinet and styletts-vc container

```
docker network connect aiavatar-network segdinet
docker network connect aiavatar-network styletts-vc
docker network inspect aiavatar-network
```


