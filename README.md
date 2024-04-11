# My First Build Project

* The built image is available from the author's [**Docker Hub** profile](https://hub.docker.com/u/oameed) and is based on [**_continuumio/miniconda3_**](https://docs.anaconda.com/free/working-with-conda/applications/docker/)  

## Cheat Sheets 

* [**_Linux Computing Cheat Sheets_**](https://archive.org/details/computing-cheat-sheets)  

* [**_GitHub Actions_**](https://docs.github.com/en/actions)  

* [**_Docker_**](https://docs.docker.com/get-started/overview/)  

  1. **_Install Docker Engine (on Ubuntu)_**  
     [follow the official instructions](https://docs.docker.com/engine/install/ubuntu/)  

  2. **_Most Used Commands_**   
  
     * **To View Images and Containers**  
       `docker image ls -a`  
       `docker container ls -a`  
  
     * **To Clean All**  
       `docker system prune -af`  
  
     * **To Run a Container**  
       `docker run <flags> <image name> <commands to run in the container>`  

       flags include: `--name` gives the container a name; `--rm` removes the container after exiting; `-it` interactive container, `--mount type=bind,source=<source>,target=<target>` makes the `<source>` (on the host) available at the `<target>` (on the container); `-d` detaches the terminal from the docker container  

     * **To Stop a Running Container**  
       `docker container stop <container ID/name>`  
     
  3. **_To Create an Image from a Dockerfile_**  
     a. `docker build -t username/image-name:image-tag <path to the Dockerfile>`  
     b. `docker login`  
     c. `docker push username/image-name:image-tag`
 
* [**_Google Cloud Platform_**](https://cloud.google.com/docs)  

  1. [**_Google Cloud Setup Checklist_**](https://cloud.google.com/docs/enterprise/setup-checklist)  

  2. **_Install the_** [**_Google Cloud CLI_**](https://cloud.google.com/sdk/docs/cheatsheet)  
     [follow the official instructions](https://cloud.google.com/sdk/docs/install-sdk) 
     



