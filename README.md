# Arsenal_Footballer_Recognation

A deep learning-based model has been developed to recognize the players of Arsenal FC team. Dataset I created a dataset using the scripts in my repository named [Arsenal_Footballer_Data_Generator](https://github.com/hcayirli97/Arsenal_Footballer_Data_Generator).

![footbalers](https://github.com/hcayirli97/Arsenal_Footballer_Recognation/blob/main/imgs/footballers.jpg)

Using [MTCNN](https://github.com/ipazc/mtcnn), the faces in the images were cropped and the faces of the football players were obtained. Training and validation images were distributed from the images of 47 football players.

![faces](https://github.com/hcayirli97/Arsenal_Footballer_Recognation/blob/main/imgs/faces.png)

After training the Resnet50 architecture with our dataset, it became available for testing. After the football players whose faces in a team photo are given as input to our trained model below, the outputs are written above the bounding boxes. In this way, it is possible to learn who the players are with a single script.

![output](https://github.com/hcayirli97/Arsenal_Footballer_Recognation/blob/main/test_images/output/output.jpg)

As you can see, the performance of our model needs to be improved. For this, more data or a different model architecture can be used.
