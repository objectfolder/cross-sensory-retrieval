# Cross-Sensory Retrieval

Cross-sensory retrieval requires the model to take one sensory modality as input and retrieve the corresponding data of another modality. For instance, given the sound of striking a mug, the "audio2vision" model needs to retrieve the corresponding image of the mug from a pool of images of hundreds of objects. In this benchmark, each sensory modality (vision, audio, touch) can be used as either input or output, leading to 9 sub-tasks.

For each object, given modality A and modality B (A and B can be either vision, touch or audio), the goal of cross-sensory retrieval is to minimize the distance between the representations of sensory observations from the same object while maximizing those from different objects. 

Specifically, we sample 100 instances from each modality of each object, resulting in two instance sets $S_A$ and $S_B$. Next, we pair the instances from both modalities, which is done by Cartesian Product:
$$
P(i)=S_A(i) \times S_B(i)
$$
, in which $i$ is the object index, and $P$ is the set of instance pairs.

## Usage

#### Data Preparation

The dataset used to train the baseline models can be downloaded from [here]([https://www.dropbox.com/scl/fo/ymd3693807jucdxj7cj1k/AMtyNZgmC1ynxFWZtVsV5gI?rlkey=hr1y85tzadepw7zb5wb9ebs0b&st=xr2keno9&dl=0](https://www.dropbox.com/scl/fo/8aus181pnr1ueiuv4jr9x/AAbHpBvpaaiuwaq4dO8hLsI?rlkey=z5atuf4urbio2c0cvx58gd9ng&st=i2q9zxdo&dl=0))

#### Training & Evaluation

Start the training process, and test the best model on test-set after training:

```sh
# Train DSCMR as an example
python main.py --model DSCMR --config_location ./configs/DSCMR.yml \
               --epochs 10 --weight_decay 1e-2 --modality_list vision touch \
               --exp DSCMR_vision_touch --batch_size 32
```

Evaluate the best model in *vision_audio_dscmr*:

```sh
# Evaluate DSCMR as an example
python main.py --model DSCMR --config_location ./configs/DSCMR.yml \
               --modality_list vision touch \
               --exp DSCMR_vision_touch --batch_size 32 \
               --eval
```

#### Add your own model

To train and test your new model on ObjectFolder Cross-Sensory Retrieval Benchmark, you only need to modify several files in *models*, you may follow these simple steps.

1. Create new model directory

    ```sh
    mkdir models/my_model
    ```

2. Design new model

    ```sh
    cd models/my_model
    touch my_model.py
    ```

3. Build the new model and its optimizer

    Add the following code into *models/build.py*:

    ```python
    elif args.model == 'my_model':
        from my_model import my_model
        model = my_model.my_model(args)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ```

4. Add the new model into the pipeline

    Once the new model is built, it can be trained and evaluated similarly:

    ```sh
    python main.py --model my_model --config_location ./configs/my_model.yml \
                   --epochs 10 --modality_list vision touch \
                   --exp my_model_vision_touch --batch_size 32
    ```

## Results on ObjectFolder Cross-Sensory Retrieval Benchmark

In our experiments, we randomly split the objects from ObjectFolder into train/val/test splits of 800/100/100 objects, and split the 10 instances of each object from ObjectFolder Real into 8/1/1. In the retrieval process, we set each instance in the input sensory modality as the query, and the instances from another sensory are retrieved by ranking them according to cosine similarity. Next, the Average Precision (AP) is computed by considering the retrieved instances from the same object as positive and others as negative. Finally, the model performance is measured by the mean Average Precision (mAP) score, which is a widely-used metric for evaluating retrieval performance.

#### Results on ObjectFolder

<table>
    <tr>
        <td>Input</td>
        <td>Retrieved</td>
        <td>RANDOM</td>
        <td>CCA</td>
        <td>PLSCA</td>
        <td>DSCMR</td>
        <td>DAR</td>
    </tr>
    <tr>
        <td rowspan="3">Vision</td>
        <td>Vision (different views)</td>
        <td>1.00</td>
        <td>55.52</td>
        <td>82.43</td>
        <td>82.74</td>
        <td>89.28</td>
    </tr>
    <tr>
        <td>Audio</td>
        <td>1.00</td>
        <td>19.56</td>
        <td>11.53</td>
        <td>9.13</td>
        <td>20.64</td>
    </tr>
    <tr>
        <td>Touch</td>
        <td>1.00</td>
        <td>6.97</td>
        <td>6.33</td>
        <td>3.57</td>
        <td>7.03</td>
    </tr>
        <tr>
        <td rowspan="3">Audio</td>
        <td>Vision</td>
        <td>1.00</td>
        <td>20.58</td>
        <td>13.37</td>
        <td>10.84</td>
        <td>20.17</td>
    </tr>
    <tr>
        <td>Audio (different vertices)</td>
        <td>1.00</td>
        <td>70.53</td>
        <td>80.77</td>
        <td>75.45</td>
        <td>77.80</td>
    </tr>
    <tr>
        <td>Touch</td>
        <td>1.00</td>
        <td>5.27</td>
        <td>6.96</td>
        <td>5.30</td>
        <td>6.91</td>
    </tr>
    </tr>
        <tr>
        <td rowspan="3">Touch</td>
        <td>Vision</td>
        <td>1.00</td>
        <td>8.50</td>
        <td>6.25</td>
        <td>4.92</td>
        <td>8.80</td>
    </tr>
    <tr>
        <td>Audio</td>
        <td>1.00</td>
        <td>6.18</td>
        <td>7.11</td>
        <td>6.15</td>
        <td>7.77</td>
    </tr>
    <tr>
        <td>Touch (different vertices)</td>
        <td>1.00</td>
        <td>28.06</td>
        <td>52.30</td>
        <td>51.08</td>
        <td>54.80</td>
    </tr>
</table>

#### Results on ObjectFolder Real

<table>
    <tr>
        <td>Input</td>
        <td>Retrieved</td>
        <td>RANDOM</td>
        <td>CCA</td>
        <td>PLSCA</td>
        <td>DSCMR</td>
        <td>DAR</td>
    </tr>
    <tr>
        <td rowspan="3">Vision</td>
        <td>Vision (different views)</td>
        <td>3.72</td>
        <td>30.60</td>
        <td>60.95</td>
        <td>81.27</td>
        <td>81.00</td>
    </tr>
    <tr>
        <td>Audio</td>
        <td>3.72</td>
        <td>12.05</td>
        <td>27.12</td>
        <td>68.34</td>
        <td>66.92</td>
    </tr>
    <tr>
        <td>Touch</td>
        <td>3.72</td>
        <td>6.29</td>
        <td>9.77</td>
        <td>64.91</td>
        <td>39.46</td>
    </tr>
        <tr>
        <td rowspan="3">Audio</td>
        <td>Vision</td>
        <td>3.72</td>
        <td>12.41</td>
        <td>30.54</td>
        <td>67.16</td>
        <td>64.35</td>
    </tr>
    <tr>
        <td>Audio (different vertices)</td>
        <td>3.72</td>
        <td>27.40</td>
        <td>55.75</td>
        <td>72.59</td>
        <td>68.79</td>
    </tr>
    <tr>
        <td>Touch</td>
        <td>3.72</td>
        <td>5.38</td>
        <td>11.66</td>
        <td>54.55</td>
        <td>33.00</td>
    </tr>
    </tr>
        <tr>
        <td rowspan="3">Touch</td>
        <td>Vision</td>
        <td>3.72</td>
        <td>6.40</td>
        <td>11.46</td>
        <td>64.86</td>
        <td>41.18</td>
    </tr>
    <tr>
        <td>Audio</td>
        <td>3.72</td>
        <td>5.57</td>
        <td>13.89</td>
        <td>55.37</td>
        <td>37.30</td>
    </tr>
    <tr>
        <td>Touch (different vertices)</td>
        <td>3.72</td>
        <td>21.16</td>
        <td>27.97</td>
        <td>66.09</td>
        <td>41.42</td>
    </tr>
</table>
