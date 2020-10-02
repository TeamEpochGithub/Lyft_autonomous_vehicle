---
title: Lyft Motion Prediction for Autonomous Vehicles (AVs)
created: '2020-09-26T22:43:21.394Z'
modified: '2020-09-30T20:43:41.212Z'
---

# Lyft Motion Prediction for Autonomous Vehicles (AVs)

## Contents
1. [Challenge Goal](#challenge_goal)
2. [Challenge Host: Lyft, Level 5](#challenge_host)
3. [Challenge Evaluation](#challenge_eval)

<!--- Challenge Goal Challenge Goal Challenge Goal  Challenge Goal  Challenge Goal -->

## Challenge Goal   <a name="challenge_goal"></a>
To build a motion prediction model to reliably predict the motion/movement of traffic agents (e.g. cars, cyclists and pedestrians) which are within an Autonomous Vehicles (AVs) environment. In short, to predict the trajectories of traficc participants (AKA agents)

<!--- Challenge Host Challenge Host Challenge Host  Challenge Host  Challenge Host -->

## Challenge Host: Lyft, Level 5     <a name="challenge_host"></a>
"The division of the ride-hailing company that is responsible for its research and development of self-driving vehicle technology (hence the name, which is borrowed from SAE’s levels of autonomy for cars, with Level 5 being the highest)."    

*SAE's*: Society of Automotive Engineer's   
<details>
<summary>For an explanation of the 6 levels of Vehicle Autonomy, expand</summary>

    - The 6 levels of Vehicle Autonomy Explained:
      - Level 0 (No Driving Automation): Manually controlled
      - Level 1 (Driver Assistance): Vehicle features a single system for driver assistance, e.g. (adaptive) cruise control. Is the lowest level of automation.
      - Level 2 (Partial Driving Automation): Means "Advanced Driver Assistance Systems" (ADAS). Tesla Autopilot and GM's, Cadillac Super Cruise systems both qualify as Level 2.
      - Level 3 (Conditional Driving Automation): Vehicles having "environmental detection" & informed decision making capabilities, e.g. accelerating past a slow-moving car. Still requires the full alertness of the driver for overriding/intervention purposes!
      - Level 4 (High Driving Automation): Key difference from Level 3 is that Level 4 AVs do not require the full alterness the driver. Level 4 AVs can operate fully in self-driving/autopilot mode b/c Level 4 AVs are capable of intervening in the event of a, e.g. potential collision or system failure. Laws, rules & regulations impose geofences (designated zones permitted for AVs) on Level 4 AVs. Most Level 4 vehicles in existence are geared toward ridesharing. Examples are:
        - NAVYA, French, builds & sells electrically powered Level 4 shuttles & cabs.
        - Alphabet's Waymo, U.S.
        - Magna, Canada
        - Volvo and Baidu strategic partnership       
        
      - Level 5 (Full Driving Automation): AVs which do not require human attention/alertness as Level 5 AVs won't even have steering wheels or acceleration/braking pedals. Level 5 AVs are free from geofencing. 
</details>

<!-- Challenge Evaluation  Challenge Evaluation  Challenge Evaluation  Challenge Evaluation  Challenge Evaluation -->


## Challenge Evaluation    <a name="challenge_eval"></a>
<ins> Recall Competition Goal:</ins> Predict the trajectories of other traffic participants.    

**Note:** "Due to the high amount of [multi-modality](https://arxiv.org/pdf/1705.09406.pdf "Multimodal Machine Learning:
A Survey and Taxonomy") and ambiguity in traffic *scenes*, the used evaluation metric to score this competition is tailored to account for multiple predictions." [Lyft Motion Prediction for Autonomous Vehicles](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/overview/evaluationm "Evaluation Page").   

<ins>Allowed to employ:</ins>
  - Uni-modal models: Yields 1 prediction per input sample
  - Multi-modal: Yields multiple hypotheses (up to 3) - further described by a confidence vector.   

<ins>Evaluation & Scoring</ins>

<!--- > **Note:** We are *encouraged* to employ/make submissions of multi-modal predictions as traffic scenes can contain a large amount of ambiguity & uncertainty. Furthermore, we are "asked to submit predictions for a private test set (no ground truth is available)" and our "solutions will be scored by Kaggle." [^1]. --->

- We are *encouraged* to employ/make submissions of multi-modal predictions as traffic scenes can contain a large amount of ambiguity & uncertainty. Furthermore, we are "asked to submit predictions for a private test set (no ground truth is available)" and our "solutions will be scored by Kaggle." [^1].

- The scoring is done by calculating the negative log-likelihood of the ground truth data given the multi-modal predictions.    


<ins>Tasks List</ins>

- [ ] Get a good, solid grasp of the [data](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/data).
    - [ ] Familiarise thy self with Lyft's Level 5 departments [L5Kit](https://github.com/lyft/l5kit) Python library.  
    - [ ] Understand Numpy [structured arrays](https://numpy.org/doc/stable/user/basics.rec.html). 
    - [ ] Understand the [zarr](https://zarr.readthedocs.io/en/stable/) data format.
    - [ ] Understand the [4 Numpy structured array types](https://github.com/lyft/l5kit/blob/master/data_format.md) `scenes`, `frames`, `agents` and `tl_faces`.
    - [ ] Understand the `World`, `Agent`, `Image`, `Satellite` and `Semantic` [coordinate systems](https://github.com/lyft/l5kit/blob/master/coords_systems.md).
    - [ ] Briefly understand [rasterization](https://en.wikipedia.org/wiki/Rasterisation)
- [ ] "Skim" through the IPython Notebook on [visualizing](https://github.com/lyft/l5kit/blob/master/examples/visualisation/visualise_data.ipynb) the data
- [ ] Actually understand the (agent motion prediction process)[https://github.com/lyft/l5kit/blob/master/examples/agent_motion_prediction/agent_motion_prediction.ipynb]

<ins>Terminologies & Concepts</ins>

<details>
<summary>For definitions, expand</summary>

- *Uni-modal*: In statistics, a unimodal (probability) distributioon is a (prob) distribution possesing a single unique *mode*, which, in this context, may refer to any peak (highest value) in the distribution.     

- *Mode*: Is the most commonly/frequently occuring value/number in a dataset. In the context of   
discrete random variables (DRV) & discrete probability distributions (DPD), the mode of a RDV, X, is the  
value, x, at which the probability mass function, PMF, of the DRV X, attains its maximum value. In short, the mode is the value that is most likely to be sampled. Lastly, numerical value of the mean (AKA expectation) parameter, $\mu$, and median, of a Normal (AKA Gaussian) (probability) distribution (NPD or GPD), are both also the same as the mode of a normal prob dist.    

- *Multi-modal*: In general, a multi-modal (probability) distribution is a probability distribution with two (bimodal dist) or more different modes. These modes appear as distinct peaks (local maximas) in a probability density function (PDF).

- *Rasterization*: The conversion/transformation process from (raw) data in vector graphic format to raster image AKA multi-channel image, e.g. an RGB image displayed on your screen 
</details>












[^1]: [Competition metrics scoring page in the L5Kit repository on GitHub](https://github.com/lyft/l5kit/blob/master/competition.md)























































<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>








<pre>asd      asd   asd   </pre>

- [x] Write the press release
- [ ] Update the website
- [ ] Contact the media

First Term
: This is the definition of the first term.

Second Term
: This is one definition of the second term.
: This is another definition of the second term.

>> zxc
> asd 
>> zxc



<pre><i>Mode</i>: Is the most commonly/frequently occuring value/number in a dataset. In the context of   
discrete random variables (DRV) & discrete probability distributions (DPD), the mode of a RDV, X, is the  
value, x, at which the probability mass function, PMF, of the DRV X, attains its maximum value.    
TL;DR:

For example, if X is a discrete random variable, then the mode is the value x (i.e. X = x) at which the probability mass function (PMF) of the discrete random, X, variable attains . In other words, it is the value that is most likely to be sampled.</pre>

