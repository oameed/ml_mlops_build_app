# My First Build Project

* The built image is available from the author's [**Docker Hub** profile](https://hub.docker.com/u/oameed) and is based on [**_continuumio/miniconda3_**](https://docs.anaconda.com/free/working-with-conda/applications/docker/)  

## Cheat Sheets 

* [**_Linux Computing Cheat Sheets_**](https://archive.org/details/computing-basics)  

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
 
## Exploring [Google Cloud Platform](https://cloud.google.com/docs)
   
   * [**_Google Cloud Setup Checklist_**](https://cloud.google.com/docs/enterprise/setup-checklist)  
   * **_Google Cloud CLI_** [[Installation]](https://cloud.google.com/sdk/docs/install-sdk) [[Cheet Sheets]](https://cloud.google.com/sdk/docs/cheatsheet)  
   * [**_MLOps on GCP_**](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)  
   * [**_Kubeflow on GCP_**](https://googlecloudplatform.github.io/kubeflow-gke-docs/dev/docs/)  
   * [**_Vertex AI_**](https://cloud.google.com/vertex-ai/docs)  

## A Tutorial on [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx)

   * [**_Simple TFX Pipeline Tutorial using Penguin Dataset_**](https://gitlab.com/oameed/ml_production_tfx)  





