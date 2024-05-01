# 5 Bird Audio Classification

You can access a working notebook on Kaggle at https://www.kaggle.com/code/collinpfeifer/5-bird-audio-classifier-model/notebook

Or you can run the notebook and front-end here.

You will need to install the following packages

```bash
pip install tensorflow numpy pandas streamlit matplotlib librosa glob
```

From there you want to go to the python notebook file titled [5-bird-audio-classifier-model.ipynb](5-bird-audio-classifier-model.ipynb)

And hit Run All at the top or run each code block individually.

It will take a loooooooong time if you cannot use a GPU. If not I reccomend using the Kaggle notebook from the above link.

To run the front-end you have to have a model saved from the time you run the model.

You can use the model we have included at model.pb, you can uncomment the last lines in the notebook and build your own.

Once that is done then do 

```bash
streamlit run streamlit.py
```

and you should be able to see your front-end running on http://localhost:3000

Make sure to upload a WAV file and see what the results are!