
# Quick readme

install python 3.11.4 and run 

```
pip3 -r requirments.txt"
```

1. To make the models and store them, 

run this code block:

```
python make_models.py
```

models are stored in 'models' folder. This should an hour or two.


2. Test the model. Two example texts are hardcoded in ['I HAVE AN OFFER FOR YOU!','Dear Maria,'], with the extra non text features in the list [26,3]
    *It should be noted that the FEATURES (non text features) are just random numbers so results won't be good. These will eventually be replaced with something meaningful.*

run this code block

```
python main.py
```

Output is a list of two probabilities for each of the two strings and two extra matching features above.
