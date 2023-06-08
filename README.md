# Cross-Sensory Retrieval

$100$ instances are sampled from each modality, resulting in two instance sets $S_A$ and $S_B$. Next, we pair the instances from both modalities, which is done by Cartesian Product:
$$
P(i)=S_A(i) \times S_B(i)
$$
, in which $i$ is the object index, and $P$ is the set of instance pairs.

For each object, given modality A and modality B (choose from vision, touch or audio), the goal of Cross-Sensory Retrieval is to minimize the distance between the representation of the instances from the same object while maximize those from different objects:
$$
\begin{cases}	\underset{\theta _1,\theta_2}{\mathrm{arg}\min}\left\{ dist\left( f_{\theta_1}\left( x \right) ,f_{\theta_2}\left( y \right) \right) \right\} , modal\left( x \right) =modal\left( y \right)\\	\underset{\theta _1,\theta_2}{\mathrm{arg}\max}\left\{ dist\left( f_{\theta_1}\left( x \right) ,f_{\theta _2}\left( y \right) \right) \right\} , modal\left( x \right) \ne modal\left( y \right)\\\end{cases}
$$

## Usage

#### Training & Evaluation

Start the training process, and test the best model on test-set after training:

```sh
cd code
# Train DSCMR as an example
python main.py --model DSCMR --config_location ./configs/DSCMR.yml \
               --modality_list vision audio \
               --exp vision_audio_dscmr
```

Evaluate the best model in *vision_audio_dscmr*:

```sh
cd code
# Evaluate DSCMR as an example
python main.py --model DSCMR --config_location ./configs/DSCMR.yml \
               --modality_list vision audio \
               --exp vision_audio_dscmr \
               --eval
```

#### Add your own model

To train and test your new model on ObjectFolder Cross-Sensory Retrieval Benchmark, you only need to modify several files in *code/models*, you may follow these simple steps.

1. Create new model directory

    ```sh
    mkdir code/models/my_model
    ```

2. Design new model

    ```sh
    cd code/models/my_model
    touch my_model.py
    ```

3. Build the new model and its optimizer

    Add the following code into *code/models/build.py*:

    ```python
    elif args.model == 'my_model':
        from my_model import my_model
        model = my_model.Encoder(args)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ```

4. Add the new model into the pipeline

    Once the new model is built, it can be trained and evaluated similarly:

    ```sh
    python main.py --model my_model --config_location ./configs/my_model.yml \
                   --modality_list vision audio \
                   --exp vision_audio_my_model \
                   --eval
    ```

## Results on ObjectFolder Cross-Sensory Retrieval Benchmark

In the experiments, the objects are split into Train: Val: Test = $8:1:1$, and each training instance is chosen from set $P$.

Several state-of-the-art methods are tested on the ObjectFolder Cross-Sensory Retrieval Benchmark. The retrieval result is measure by mean Average Precision (mAP): each instance is set as the input respectively, and other instances are retrieved and ranked using the cosine distance.

<table>
    <tr>
        <td>Input</td>
        <td>Retrieved</td>
        <td>CCA</td>
        <td>KCCA</td>
        <td>DCCA</td>
        <td>DAR</td>
        <td>DSCMR</td>
    </tr>
    <tr>
        <td rowspan="3">Vision</td>
        <td>Vision (different views)</td>
        <td>40.33</td>
        <td>7.49</td>
        <td>8.78</td>
        <td>73.62</td>
        <td>77.94</td>
    </tr>
    <tr>
        <td>Audio</td>
        <td>9.49</td>
        <td>9.23</td>
        <td>7.45</td>
        <td>30.64</td>
        <td>30.70</td>
    </tr>
    <tr>
        <td>Touch</td>
        <td>6.75</td>
        <td>4.93</td>
        <td>3.75</td>
        <td>13.59</td>
        <td>12.38</td>
    </tr>
        <tr>
        <td rowspan="3">Audio</td>
        <td>Vision</td>
        <td>8.75</td>
        <td>8.03</td>
        <td>7.15</td>
        <td>27.04</td>
        <td>31.00</td>
    </tr>
    <tr>
        <td>Audio (different vertices)</td>
        <td>28.18</td>
        <td>19.70</td>
        <td>9.05</td>
        <td>76.33</td>
        <td>85.30</td>
    </tr>
    <tr>
        <td>Touch</td>
        <td>5.58</td>
        <td>4.28</td>
        <td>4.16</td>
        <td>7.80</td>
        <td>9.00</td>
    </tr>
    </tr>
        <tr>
        <td rowspan="3">Touch</td>
        <td>Vision</td>
        <td>7.29</td>
        <td>5.08</td>
        <td>4.81</td>
        <td>13.02</td>
        <td>13.80</td>
    </tr>
    <tr>
        <td>Audio</td>
        <td>5.95</td>
        <td>6.66</td>
        <td>6.18</td>
        <td>17.88</td>
        <td>15.43</td>
    </tr>
    <tr>
        <td>Touch (different vertices)</td>
        <td>15.00</td>
        <td>9.16</td>
        <td>4.03</td>
        <td>37.11</td>
        <td>31.41</td>
    </tr>
</table>

