# Install the dependencies

```
pip install -r requirements.txt
```

# Train Lapped Transforms on the both the set of images
```
python src/train.py
```

# Do the transformations for both the set of images 
```
python src/main.py --type colored
python src/main.py --type gray
```

# Visualize plots
```
python src/visualization.py --type colored
python src/visualization.py --type gray
```