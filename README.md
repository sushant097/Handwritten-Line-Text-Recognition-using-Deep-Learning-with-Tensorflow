# Handwritten Line Text Recognition using Deep Learning with Tensorflow
### Description
Use Convolutional Recurrent Neural Network to recognize the Handwritten line text image without pre segmentation into words or characters. Use CTC loss Function to train.

Special Thanks for the [SimpleHTR](https://github.com/githubharald/SimpleHTR) and [Lamhoangtung](https://github.com/lamhoangtung/LineHTR) for the great contribution.

### Why Deep Learning?
![Why Deep Learning](images/WhyDeepLearning.png?raw=true "Why Deep Learning")
> Deep Learning self extracts features with a deep neural networks and classify itself. Compare to traditional Algorithms it performance increase with Amount of Data.

## <i> Basic Intuition on How it Works.
![Step_wise_detail](images/Step_wise_detail_of_workflow.png?raw=true "Step_Wise Detail")
* First Use Convolutional Recurrent Neural Network to extract the important features from the handwritten line text Image.
* The output before CNN FC layer (512x100x8) is passed to the BLSTM which is for sequence dependency and time-sequence operations.
* Then CTC LOSS [Alex Graves](https://www.cs.toronto.edu/~graves/icml_2006.pdf) is used to train the RNN which eliminate the Alignment problem in Handwritten, since handwritten have different alignment of every writers. We just gave the what is written in the image (Ground Truth Text) and BLSTM output, then it calculates loss simply as -log("gtText"); aim to minimize negative maximum likelihood path.See [this](https://distill.pub/2017/ctc/) for detail. 
* Finally CTC finds out the possible paths from the given labels. Loss is given by for (X,Y) pair is: ![Ctc_Loss](images/CtcLossFormula.png?raw=true "CTC loss for the (X,Y) pair")
* Finally CTC Decode is used to decode the output during Prediction.
</i>


#### Detail Project Workflow
![Architecture of Model](images/ArchitectureDetails.png?raw=true "Model Architecture")

# Requirements
1. Tensorflow 1.8.0
2. Flask
3. Numpy
4. OpenCv 3
5. Spell Checker `autocorrect` >=0.3.0 ``pip install autocorrect``

#### Dataset Used
* IAM dataset download from [here](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
* Only needed the lines images and lines.txt (ASCII).
* Place the downloaded files inside data directory  

###### You can find trained model to download from [here.](https://drive.google.com/open?id=10HHNZPqPQZCQCLrKGQOq5E7zFW5wGcA4) Download and extract all files inside the `model/` directory.


To Train the model from scratch
```markdown
$ python main.py --train

```
To validate the model
```markdown
$ python main.py --validate
```
To Prediction
```markdown
$ python main.py
```

Run in Web with Flask
```markdown
$ python upload.py
Validation character error rate of saved model: 8.654728%
Python: 3.6.4 
Tensorflow: 1.8.0
Init with stored values from ../model/snapshot-24
Without Correction clothed leaf by leaf with the dioappoistmest
With Correction clothed leaf by leaf with the dioappoistmest
```
**Prediction output on IAM Test Data**
![PredictionOutput](images/IAM_dataset_Prediction_Output.png?raw=true "Prediction Output On Iam Dataset")

**Prediction output on Self Test Data**
![PredictionOutput](images/PredictionOutput.png?raw=true "Prediction Output on Self Data")

**The Validation character error rate of saved model: 8.654728%**
# Further Improvement
* Line segementation can be added for full paragraph text recognition
* Better Image preprocessing to handle real time image.
* Better Decoding approach to improve accuracy.
* More variety of data for real time recognition.
