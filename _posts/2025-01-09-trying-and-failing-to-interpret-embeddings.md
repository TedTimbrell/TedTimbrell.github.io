---
layout: post
title:  "Trying and failing to interpret sentence embeddings"
date:   2025-01-09 11:53:40 -0800
categories: AI
---


I was born with congenital anosmia i.e. I cannot and have never been able to smell. Farts, flowers, cookies, and perfume; I have no personal experience of any of these smells. Yet, I can tell you that farts take over a room and cookies smell like home. This is all picked up through context. It's picked up from watching my friends retching at the stench of the small animal that died in the vents of my middle school. For me, a smell is defined by its relation to other smells and the emotive descriptions of others. This is not altogether different from a sentence embedding.

I want to eventually build a system that can help me interpret smells. Now, I know I could ask an LLM (or a friend) to describe the smell of something but I do wonder if vector addition could provide some unexpected insights. What smells are quite similar but distant in context? I'd also like to try using reduced vectors to generate music or some other synesthetic output.


In this post, I'm going to explore vector addition and vector rotations as a means of modifying and interpreting these embeddings. My explorations are (mostly) a failure although hopefully, my process might save someone else some time. 

If you have any ideas or corrections, please email me at `ted@timbrell.dev`


## Background
My inspiration for this comes from [_Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings_](https://arxiv.org/pdf/1607.06520). Except in this case, I'm using sentence embeddings rather than word embeddings. If I want to embed smells, I'll need to be able to input "the smell of red wine."

```python
from openai import OpenAI
import numpy as np
import heapq
import pandas as pd
import os


openai_client = OpenAI()


def get_embedding_openai(names):
    response = openai_client.embeddings.create(
        input=names, model="text-embedding-3-small"
    )
    return np.array([d.embedding for d in response.data])

```


```python
king, queen, man, woman, prince, princess = get_embedding_openai(
    [
        "king of england",
        "queen of england",
        "man",
        "woman",
        "prince of england",
        "princess of england",
    ]
)
son, daughter, actor, actress, steward, stewardess = get_embedding_openai([
    "son",
    "daughter",
    "actor",
    "actress",
    "steward",
    "stewardess",
])
```

> Q: Wait aren't you supposed to be using smells?

It's pretty hard to reason about something you can't experience. I might know fresh coffee smells good in the morning but I can't tell how similar/dissimilar that is the smell of dew in the morning. I'll get to smells in a later post. 

I'm already making a jump from word embeddings to sentence embeddings so I believe it's worth revisitng gender before moving on. It also has the benefit of being easy to generate examples for and is built into the English language. I'll be using cosine similarity and Euclidean distance to get a sense of the distance between the vectors.


```python
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0

    return dot_product / (magnitude_vec1 * magnitude_vec2)


def euc_dist(a, b):
    return sum(abs(a - b))
```

### Simple example: vector offsets and addition
Let's try to get the vector for "King" from the vector for "Queen".


```python
male_offset = man - woman
added_queen = queen + male_offset
print(f"{cosine_similarity(king, queen)=}")
print(f"{cosine_similarity(king, added_queen)=}")
```

    cosine_similarity(king, queen)=np.float64(0.7561968293567973)
    cosine_similarity(king, added_queen)=np.float64(0.7436281583952487)



```python
print(f"{euc_dist(king, queen)=}")
print(f"{euc_dist(king, added_queen)=}")
```

    euc_dist(king, queen)=np.float64(21.68610072977549)
    euc_dist(king, added_queen)=np.float64(23.88454184561374)


Well, that's annoying. Unlike what I'd expect from the word embedding paper, our vector for "Queen" plus the gender offset is further away from the vector for "King" both in angle and Euclidean distance.

I'm also surprised by just how little the similarity metrics moved. Then again, the geometry is unclear here. The vector offset might be going in the wrong direction or under/overshooting.


```python
f"man - woman offset magnitude: {np.linalg.norm(male_offset)}", f"King - queen offset magnitude {np.linalg.norm(king - queen)}"
```




    ('man - woman offset magnitude: 0.7648199511941038',
     'King - queen offset magnitude 0.6982881772713763')



So we're moving, roughly, the same distance as we'd need to reach the "king" vector.


```python
f"{np.arccos(cosine_similarity(added_queen, queen))} radians between added_queen and queen"
f"{np.arccos(cosine_similarity(king, queen))} radians between king and queen"
```




    '0.7318444725115928 radians between added_queen and queen'
    '0.7133150836556438 radians between king and queen'



And we're changing our angle by roughly the same amount as expected... just not in the right direction.

### Let's take a look at the cosine similarity between these gendered offsets


```python
gender_vectors = [
    man - woman,
    king - queen,
    prince - princess,
    son - daughter,
    actor - actress,
    steward - stewardess,
]
for idx in range(len(gender_vectors)):
    gender_vectors[idx] /= np.linalg.norm(gender_vectors[0])

res = np.zeros(shape=(len(gender_vectors), len(gender_vectors)))

for r in range(len(gender_vectors)):
    for c in range(len(gender_vectors)):
        res[r, c] = cosine_similarity(gender_vectors[r], gender_vectors[c])
pd.DataFrame(res)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.455847</td>
      <td>0.438728</td>
      <td>0.244890</td>
      <td>0.276235</td>
      <td>0.214461</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.455847</td>
      <td>1.000000</td>
      <td>0.657469</td>
      <td>0.222574</td>
      <td>0.386973</td>
      <td>0.355544</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.438728</td>
      <td>0.657469</td>
      <td>1.000000</td>
      <td>0.229321</td>
      <td>0.337566</td>
      <td>0.385467</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.244890</td>
      <td>0.222574</td>
      <td>0.229321</td>
      <td>1.000000</td>
      <td>0.154418</td>
      <td>0.111242</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.276235</td>
      <td>0.386973</td>
      <td>0.337566</td>
      <td>0.154418</td>
      <td>1.000000</td>
      <td>0.232568</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.214461</td>
      <td>0.355544</td>
      <td>0.385467</td>
      <td>0.111242</td>
      <td>0.232568</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Despite the thought that these are just gendered versions of the same concept... the offsets point in quite different directions. `son - daughter` differs from `steward - stewardess` by 1.47 radians (or 84 degrees).

## Rotation
I'm not up to date on research into embeddings but I find the use of vector addition for these analyses odd. I know that these vectors are generated through a series of additions and activations but if these models are normalizing everything to a unit vector and comparing everything with cosine similarity are we not inherently saying that it's the angles that matter?

To that end, what if I rotate the "queen" vector along the plane created by the "man" and "woman" vectors? The vectors for "king" and "queen" have to be offset from our vectors for "man" and "woman". We also know that embeddings capture more concepts than just the N dimensions represented in the vector. A rotation, while more expensive, could help in the case of an angular difference between the initial vector pair and the compared vector pair.

Below, we try rotating our queen vector with the rotation matrix found from getting to "man" from "woman".


```python
def compute_nd_rotation_matrix(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    
    cos_theta = np.dot(a_norm, b_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    
    v = b_norm - np.dot(b_norm, a_norm) * a_norm
    v_norm = np.linalg.norm(v)
    
    if v_norm < 1e-8:  # a and b are collinear
        return np.eye(len(a)),
    
    v = v / v_norm
    
    identity = np.eye(len(a))
    outer_aa = np.outer(a_norm, a_norm)
    outer_av = np.outer(a_norm, v)
    outer_va = np.outer(v, a_norm)
    outer_vv = np.outer(v, v)
    
    R = (
        identity
        + np.sin(angle) * (outer_va - outer_av)
        + (np.cos(angle) - 1) * (outer_vv + outer_aa)
    )
    
    return R, angle
    

gender_rotation, gender_angle = compute_nd_rotation_matrix(woman, man)

rotated_queen = np.dot(gender_rotation, queen)
```


```python
def highlight_max(s):
    is_max = s == s.max()
    return ["font-weight: bold" if v else "" for v in is_max]


def highlight_min(s):
    is_min = s == s.min()
    return ["font-weight: bold" if v else "" for v in is_min]


def compute_results(*, target, source, offset, rotation):
    target_norm = target / np.linalg.norm(target)
    source_norm = source / np.linalg.norm(source)
    added_source = source_norm + offset
    added_source /= np.linalg.norm(added_source)

    rotated_source = np.dot(rotation, source_norm)

    rotated_vector_metrics = {
        "cosine_similarity": cosine_similarity(target_norm, rotated_source),
        "euclidean_distance": euc_dist(target_norm, rotated_source),
    }

    summed_vector_metrics = {
        "cosine_similarity": cosine_similarity(target_norm, added_source),
        "euclidean_distance": euc_dist(target_norm, added_source),
    }

    original_vector_metrics = {
        "cosine_similarity": cosine_similarity(target_norm, source_norm),
        "euclidean_distance": euc_dist(target_norm, source_norm),
    }

    df = pd.DataFrame(
        {
            "Original Vector": original_vector_metrics,
            "Summed Vector": summed_vector_metrics,
            "Rotated Vector": rotated_vector_metrics,
        }
    ).T
    return df


def style_results(df):
    styled_df = df.style.apply(highlight_max, subset=["cosine_similarity"])
    styled_df.apply(highlight_min, subset=["euclidean_distance"])
    return styled_df


style_results(
    compute_results(
        target=king, source=queen, offset=male_offset, rotation=gender_rotation
    )
)
```




<style type="text/css">
#T_c246d_row2_col0, #T_c246d_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_c246d">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_c246d_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_c246d_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c246d_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_c246d_row0_col0" class="data row0 col0" >0.756197</td>
      <td id="T_c246d_row0_col1" class="data row0 col1" >21.686100</td>
    </tr>
    <tr>
      <th id="T_c246d_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_c246d_row1_col0" class="data row1 col0" >0.743628</td>
      <td id="T_c246d_row1_col1" class="data row1 col1" >22.333814</td>
    </tr>
    <tr>
      <th id="T_c246d_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_c246d_row2_col0" class="data row2 col0" >0.800727</td>
      <td id="T_c246d_row2_col1" class="data row2 col1" >19.759377</td>
    </tr>
  </tbody>
</table>





```python
np.arccos(0.756197)- np.arccos(0.800727)
```




    np.float64(0.07102636138332474)



The rotation helps! Though, it only moves us 0.07 radians (4 degrees) closer. 

### Let's try with other gendered titles.

Below we try with "prince" and "princess",


```python

style_results(compute_results(target=prince, source=princess, rotation=gender_rotation, offset=male_offset))
```




<style type="text/css">
#T_c0d8b_row2_col0, #T_c0d8b_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_c0d8b">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_c0d8b_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_c0d8b_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c0d8b_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_c0d8b_row0_col0" class="data row0 col0" >0.798122</td>
      <td id="T_c0d8b_row0_col1" class="data row0 col1" >19.734372</td>
    </tr>
    <tr>
      <th id="T_c0d8b_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_c0d8b_row1_col0" class="data row1 col0" >0.752733</td>
      <td id="T_c0d8b_row1_col1" class="data row1 col1" >21.830848</td>
    </tr>
    <tr>
      <th id="T_c0d8b_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_c0d8b_row2_col0" class="data row2 col0" >0.831910</td>
      <td id="T_c0d8b_row2_col1" class="data row2 col1" >17.951380</td>
    </tr>
  </tbody>
</table>




This yields similar results, although this is just another title for royalty.


```python
style_results(compute_results(target=son, source=daughter, rotation=gender_rotation, offset=male_offset))
```




<style type="text/css">
#T_bae64_row2_col0, #T_bae64_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_bae64">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_bae64_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_bae64_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_bae64_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_bae64_row0_col0" class="data row0 col0" >0.506902</td>
      <td id="T_bae64_row0_col1" class="data row0 col1" >30.924019</td>
    </tr>
    <tr>
      <th id="T_bae64_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_bae64_row1_col0" class="data row1 col0" >0.505972</td>
      <td id="T_bae64_row1_col1" class="data row1 col1" >31.099625</td>
    </tr>
    <tr>
      <th id="T_bae64_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_bae64_row2_col0" class="data row2 col0" >0.519920</td>
      <td id="T_bae64_row2_col1" class="data row2 col1" >30.628173</td>
    </tr>
  </tbody>
</table>





```python
style_results(compute_results(target=actor, source=actress, rotation=gender_rotation, offset=male_offset))
```




<style type="text/css">
#T_8fd38_row2_col0, #T_8fd38_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_8fd38">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_8fd38_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_8fd38_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_8fd38_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_8fd38_row0_col0" class="data row0 col0" >0.618884</td>
      <td id="T_8fd38_row0_col1" class="data row0 col1" >27.222238</td>
    </tr>
    <tr>
      <th id="T_8fd38_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_8fd38_row1_col0" class="data row1 col0" >0.565936</td>
      <td id="T_8fd38_row1_col1" class="data row1 col1" >29.213596</td>
    </tr>
    <tr>
      <th id="T_8fd38_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_8fd38_row2_col0" class="data row2 col0" >0.644523</td>
      <td id="T_8fd38_row2_col1" class="data row2 col1" >26.165533</td>
    </tr>
  </tbody>
</table>





```python
style_results(compute_results(target=steward, source=stewardess, rotation=gender_rotation, offset=male_offset))
```




<style type="text/css">
#T_2389b_row2_col0, #T_2389b_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_2389b">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_2389b_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_2389b_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_2389b_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_2389b_row0_col0" class="data row0 col0" >0.753096</td>
      <td id="T_2389b_row0_col1" class="data row0 col1" >21.497546</td>
    </tr>
    <tr>
      <th id="T_2389b_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_2389b_row1_col0" class="data row1 col0" >0.633975</td>
      <td id="T_2389b_row1_col1" class="data row1 col1" >26.397621</td>
    </tr>
    <tr>
      <th id="T_2389b_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_2389b_row2_col0" class="data row2 col0" >0.760474</td>
      <td id="T_2389b_row2_col1" class="data row2 col1" >21.201120</td>
    </tr>
  </tbody>
</table>




I have two takeaways here, 1) that the summed vector is _worse_ in every pairing and 2) that the rotated vector is encoding some aspect of gender (but the improvement is quite small). Let's explore each of these.

#### 1. The summed vector is further from the target than the original vector of all title pairs.
I find this result suspicious. Let's try scaling the offset vector to see if I can get a better result.


```python
from scipy.optimize import minimize

def objective(k, source, offset, target):
    adjusted_vector = source + k * offset
    return -cosine_similarity(adjusted_vector, target)

options = []
for target, source in [
    (king, queen),
    (prince, princess),
    (son, daughter),
    (actor, actress),
    (steward, stewardess),
]:
    initial_k = 0.0

    result = minimize(objective, initial_k, args=(source, male_offset, target))
    optimal_k = result.x[0]
    options.append(optimal_k)
    print(optimal_k)
average_k = sum(options) / len(options)
print(f"Average K: {average_k}")
    
```

    0.4442216918664982
    0.3758034405295738
    0.45508087793089264
    0.3248524288805894
    0.18004962701645405
    Average K: 0.3560016132448016


Above, I'm printing the individual best-fit scalar modifier for our gender offset vector for each pair. We can see it's overshooting in every case. 

For simplicity, let's average the optimal scalar and recompute the similarity stats. This is not optimal as I should be minimizing on the batch and then testing on out-of-sample data. 

Trivial to say; that a singular, consistent magnitude for the offset would have been nice. In the case where we don't have a known target, that offset would allow us to naively add/subtract the gender offset to a source vector and have confidence in its meaning.


```python
for target, source in [
    (king, queen),
    (prince, princess),
    (son, daughter),
    (actor, actress),
    (steward, stewardess),
]:
    display(style_results(compute_results(target=target, source=source, rotation=gender_rotation, offset=male_offset * average_k)))
    print()
```


<style type="text/css">
#T_cbec9_row1_col0, #T_cbec9_row1_col1 {
  font-weight: bold;
}
</style>
<table id="T_cbec9">
  <thead>
    <tr>
      <th class="blank level0" >Queen -> King</th>
      <th id="T_cbec9_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_cbec9_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_cbec9_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_cbec9_row0_col0" class="data row0 col0" >0.756197</td>
      <td id="T_cbec9_row0_col1" class="data row0 col1" >21.686100</td>
    </tr>
    <tr>
      <th id="T_cbec9_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_cbec9_row1_col0" class="data row1 col0" >0.801380</td>
      <td id="T_cbec9_row1_col1" class="data row1 col1" >19.744261</td>
    </tr>
    <tr>
      <th id="T_cbec9_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_cbec9_row2_col0" class="data row2 col0" >0.800727</td>
      <td id="T_cbec9_row2_col1" class="data row2 col1" >19.759377</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_ad0b7_row1_col0, #T_ad0b7_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_ad0b7">
  <thead>
    <tr>
      <th class="blank level0" >Princess -> Prince</th>
      <th id="T_ad0b7_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_ad0b7_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_ad0b7_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_ad0b7_row0_col0" class="data row0 col0" >0.798122</td>
      <td id="T_ad0b7_row0_col1" class="data row0 col1" >19.734372</td>
    </tr>
    <tr>
      <th id="T_ad0b7_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_ad0b7_row1_col0" class="data row1 col0" >0.833044</td>
      <td id="T_ad0b7_row1_col1" class="data row1 col1" >18.017618</td>
    </tr>
    <tr>
      <th id="T_ad0b7_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_ad0b7_row2_col0" class="data row2 col0" >0.831910</td>
      <td id="T_ad0b7_row2_col1" class="data row2 col1" >17.951380</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_86d07_row1_col0, #T_86d07_row1_col1 {
  font-weight: bold;
}
</style>
<table id="T_86d07">
  <thead>
    <tr>
      <th class="blank level0" >Daughter -> Son</th>
      <th id="T_86d07_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_86d07_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_86d07_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_86d07_row0_col0" class="data row0 col0" >0.506902</td>
      <td id="T_86d07_row0_col1" class="data row0 col1" >30.924019</td>
    </tr>
    <tr>
      <th id="T_86d07_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_86d07_row1_col0" class="data row1 col0" >0.537458</td>
      <td id="T_86d07_row1_col1" class="data row1 col1" >29.976789</td>
    </tr>
    <tr>
      <th id="T_86d07_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_86d07_row2_col0" class="data row2 col0" >0.519920</td>
      <td id="T_86d07_row2_col1" class="data row2 col1" >30.628173</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_5b209_row2_col0, #T_5b209_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_5b209">
  <thead>
    <tr>
      <th class="blank level0" >Actress -> Actor</th>
      <th id="T_5b209_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_5b209_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_5b209_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_5b209_row0_col0" class="data row0 col0" >0.618884</td>
      <td id="T_5b209_row0_col1" class="data row0 col1" >27.222238</td>
    </tr>
    <tr>
      <th id="T_5b209_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_5b209_row1_col0" class="data row1 col0" >0.638717</td>
      <td id="T_5b209_row1_col1" class="data row1 col1" >26.398096</td>
    </tr>
    <tr>
      <th id="T_5b209_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_5b209_row2_col0" class="data row2 col0" >0.644523</td>
      <td id="T_5b209_row2_col1" class="data row2 col1" >26.165533</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_0807d_row2_col0, #T_0807d_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_0807d">
  <thead>
    <tr>
      <th class="blank level0" >Stewardess -> Steward</th>
      <th id="T_0807d_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_0807d_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0807d_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_0807d_row0_col0" class="data row0 col0" >0.753096</td>
      <td id="T_0807d_row0_col1" class="data row0 col1" >21.497546</td>
    </tr>
    <tr>
      <th id="T_0807d_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_0807d_row1_col0" class="data row1 col0" >0.753192</td>
      <td id="T_0807d_row1_col1" class="data row1 col1" >21.533765</td>
    </tr>
    <tr>
      <th id="T_0807d_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_0807d_row2_col0" class="data row2 col0" >0.760474</td>
      <td id="T_0807d_row2_col1" class="data row2 col1" >21.201120</td>
    </tr>
  </tbody>
</table>



    


There we go! Our summed vector is now better or at least matches our original vector's similarity. The summed vector now also matches the performance of the rotated vector, although it required an additional optimization step and K chosen in-sample. 

Our largest outlier pair when optimizing for our scalar K was "steward" and "stewardess". The optimal scalar K for that pair is half the average. Still, we the addition does no harm in terms of distance from the target. Though, we see the rotated vector makes progress in approaching the target vector.

I recognize that only using the "man" and "woman" vectors to generate the offset is a bit silly. Using a broader collection of gendered words, sentences, titles, etc., and averaging them to create average "man" and "woman" vectors before taking the offset is best practice. However, I'm going to have limited data once I get to smells so I'm trying to keep this simple.

Now that I've resolved the issue with vector addition, let's go back to rotations!

#### 2. The rotated vector is closer to the target! 
The rotated vector is helping a little but isn't closing much of the gap between the vectors. As we saw earlier, we're rotating by roughly the correct amount, but on the wrong plane. 

I _think_ I'm encoding some concept of gender in the rotation but am I accounting for it entirely? While I might say there are no differences between a King and a Queen the studies on bias in LLMs show us that isn't the case. I expect some difference in the transformed vectors no matter what naive transformations are performed. Gender stereotypes encoded into the embedding of "prince of England" might not be encoded into the general embedding for "man" (or are lessened through averaging). So, is the remaining distance due to other features/meanings or am I failing to account for general aspects of gender? 

To start with, let's make this a fair comparison with the offset and optimize the angle (magnitude) of rotation. After all, my hypothesis is that the two-dimensional plane can be treated as a feature, and its angle as a magnitude.

A scalar product for our angle wouldn't be all that interpretable so instead, I'll optimize for the angle of rotation directly.


```python
def compute_nd_rotation_matrix(a, b, angle=None):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    
    cos_theta = np.dot(a_norm, b_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    if angle is None:
        angle = np.arccos(cos_theta)
    
    v = b_norm - np.dot(b_norm, a_norm) * a_norm
    v_norm = np.linalg.norm(v)
    
    if v_norm < 1e-8:  # a and b are collinear
        return np.eye(len(a)),
    
    v = v / v_norm
    
    identity = np.eye(len(a))
    outer_aa = np.outer(a_norm, a_norm)
    outer_av = np.outer(a_norm, v)
    outer_va = np.outer(v, a_norm)
    outer_vv = np.outer(v, v)
    
    R = (
        identity
        + np.sin(angle) * (outer_va - outer_av)
        + (np.cos(angle) - 1) * (outer_vv + outer_aa)
    )
    
    return R, angle


def objective(m, source, base_source, base_target , target):
    R, angle = compute_nd_rotation_matrix(base_source, base_target, m)    
    adjusted_vector = np.dot(R, source)
    return -cosine_similarity(adjusted_vector, target)

options = []
for target, source in [
    (king, queen),
    (prince, princess),
    (son, daughter),
    (actor, actress),
    (steward, stewardess),
]:
    initial_m = gender_angle

    result = minimize(objective, initial_m, args=(source, woman, man, target))
    optimal_m = result.x[0]
    options.append(optimal_m)
    print(optimal_m)
print()
average_m = sum(options) / len(options)
print(f"Average Optimized Angle: {average_m} (radians)")
print(f"Orginal Angle: {gender_angle} (radians)")
print(f"Difference: {abs(average_m - gender_angle)} radians, {np.rad2deg(abs(average_m - gender_angle))} degrees")
```

    1.1203259132974637
    1.0205728689773772
    0.4497845291476582
    0.5237429339776948
    0.4732922638975175
    
    Average Optimized Angle: 0.7175437018595423 (radians)
    Orginal Angle: 0.7848061954720583 (radians)
    Difference: 0.06726249361251602 radians, 3.8538570035228257 degrees


For idiots like myself, this makes the range for the optimal angle \[26, 63\] degrees. A lot like our offset, it would have been nice if this range was small. 

The angle between the "man" and "woman" vectors is quite similar to the average of our optimized angles on our pairs! The average optimized angle is 90% of the original angle (only three degrees off!) whereas the optimized offset magnitude is 34% of the original magnitude. If this were to hold for concepts other than gender, rotations might be easier to work with.

#### Let's check how the optimized similarity metrics performed with our optimized angle...


```python
for target, source in [
    (king, queen),
    (prince, princess),
    (son, daughter),
    (actor, actress),
    (steward, stewardess),
]:
    optimized_rotation_df = compute_results(
        target=target, 
        source=source, 
        rotation=compute_nd_rotation_matrix(woman, man, average_m)[0], 
        offset=male_offset * average_k
    ).T
    df = compute_results(
        target=target, 
        source=source, 
        rotation=gender_rotation, 
        offset=male_offset * average_k
    ).T
    df["Optimized Rotated Vector"] = optimized_rotation_df["Rotated Vector"]
    display(style_results(df.T))
    print()
```


<style type="text/css">
#T_c3916_row1_col0, #T_c3916_row1_col1 {
  font-weight: bold;
}
</style>
<table id="T_c3916">
  <thead>
    <tr>
      <th class="blank level0" >Queen -> King</th>
      <th id="T_c3916_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_c3916_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c3916_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_c3916_row0_col0" class="data row0 col0" >0.756197</td>
      <td id="T_c3916_row0_col1" class="data row0 col1" >21.686100</td>
    </tr>
    <tr>
      <th id="T_c3916_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_c3916_row1_col0" class="data row1 col0" >0.801380</td>
      <td id="T_c3916_row1_col1" class="data row1 col1" >19.744261</td>
    </tr>
    <tr>
      <th id="T_c3916_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_c3916_row2_col0" class="data row2 col0" >0.800727</td>
      <td id="T_c3916_row2_col1" class="data row2 col1" >19.759377</td>
    </tr>
    <tr>
      <th id="T_c3916_level0_row3" class="row_heading level0 row3" >Optimized Rotated Vector</th>
      <td id="T_c3916_row3_col0" class="data row3 col0" >0.798603</td>
      <td id="T_c3916_row3_col1" class="data row3 col1" >19.853226</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_18967_row1_col0, #T_18967_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_18967">
  <thead>
    <tr>
      <th class="blank level0" >Princess -> Prince</th>
      <th id="T_18967_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_18967_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_18967_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_18967_row0_col0" class="data row0 col0" >0.798122</td>
      <td id="T_18967_row0_col1" class="data row0 col1" >19.734372</td>
    </tr>
    <tr>
      <th id="T_18967_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_18967_row1_col0" class="data row1 col0" >0.833044</td>
      <td id="T_18967_row1_col1" class="data row1 col1" >18.017618</td>
    </tr>
    <tr>
      <th id="T_18967_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_18967_row2_col0" class="data row2 col0" >0.831910</td>
      <td id="T_18967_row2_col1" class="data row2 col1" >17.951380</td>
    </tr>
    <tr>
      <th id="T_18967_level0_row3" class="row_heading level0 row3" >Optimized Rotated Vector</th>
      <td id="T_18967_row3_col0" class="data row3 col0" >0.830565</td>
      <td id="T_18967_row3_col1" class="data row3 col1" >18.004223</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_e6c1f_row1_col0, #T_e6c1f_row1_col1 {
  font-weight: bold;
}
</style>
<table id="T_e6c1f">
  <thead>
    <tr>
      <th class="blank level0" >Daughter -> Son</th>
      <th id="T_e6c1f_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_e6c1f_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e6c1f_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_e6c1f_row0_col0" class="data row0 col0" >0.506902</td>
      <td id="T_e6c1f_row0_col1" class="data row0 col1" >30.924019</td>
    </tr>
    <tr>
      <th id="T_e6c1f_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_e6c1f_row1_col0" class="data row1 col0" >0.537458</td>
      <td id="T_e6c1f_row1_col1" class="data row1 col1" >29.976789</td>
    </tr>
    <tr>
      <th id="T_e6c1f_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_e6c1f_row2_col0" class="data row2 col0" >0.519920</td>
      <td id="T_e6c1f_row2_col1" class="data row2 col1" >30.628173</td>
    </tr>
    <tr>
      <th id="T_e6c1f_level0_row3" class="row_heading level0 row3" >Optimized Rotated Vector</th>
      <td id="T_e6c1f_row3_col0" class="data row3 col0" >0.525843</td>
      <td id="T_e6c1f_row3_col1" class="data row3 col1" >30.419578</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_1ca2e_row3_col0, #T_1ca2e_row3_col1 {
  font-weight: bold;
}
</style>
<table id="T_1ca2e">
  <thead>
    <tr>
      <th class="blank level0" >Actress -> Actor</th>
      <th id="T_1ca2e_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_1ca2e_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1ca2e_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_1ca2e_row0_col0" class="data row0 col0" >0.618884</td>
      <td id="T_1ca2e_row0_col1" class="data row0 col1" >27.222238</td>
    </tr>
    <tr>
      <th id="T_1ca2e_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_1ca2e_row1_col0" class="data row1 col0" >0.638717</td>
      <td id="T_1ca2e_row1_col1" class="data row1 col1" >26.398096</td>
    </tr>
    <tr>
      <th id="T_1ca2e_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_1ca2e_row2_col0" class="data row2 col0" >0.644523</td>
      <td id="T_1ca2e_row2_col1" class="data row2 col1" >26.165533</td>
    </tr>
    <tr>
      <th id="T_1ca2e_level0_row3" class="row_heading level0 row3" >Optimized Rotated Vector</th>
      <td id="T_1ca2e_row3_col0" class="data row3 col0" >0.648405</td>
      <td id="T_1ca2e_row3_col1" class="data row3 col1" >26.021171</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_f8158_row3_col0, #T_f8158_row3_col1 {
  font-weight: bold;
}
</style>
<table id="T_f8158">
  <thead>
    <tr>
      <th class="blank level0" >Stewardess -> Steward</th>
      <th id="T_f8158_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_f8158_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f8158_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_f8158_row0_col0" class="data row0 col0" >0.753096</td>
      <td id="T_f8158_row0_col1" class="data row0 col1" >21.497546</td>
    </tr>
    <tr>
      <th id="T_f8158_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_f8158_row1_col0" class="data row1 col0" >0.753192</td>
      <td id="T_f8158_row1_col1" class="data row1 col1" >21.533765</td>
    </tr>
    <tr>
      <th id="T_f8158_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_f8158_row2_col0" class="data row2 col0" >0.760474</td>
      <td id="T_f8158_row2_col1" class="data row2 col1" >21.201120</td>
    </tr>
    <tr>
      <th id="T_f8158_level0_row3" class="row_heading level0 row3" >Optimized Rotated Vector</th>
      <td id="T_f8158_row3_col0" class="data row3 col0" >0.762675</td>
      <td id="T_f8158_row3_col1" class="data row3 col1" >21.091211</td>
    </tr>
  </tbody>
</table>



    


Welp, `small_rotation == small_similarity_change`, is hardly surprising. 

Similarity improves in the three pairs with a smaller optimal rotation angle and loses in the two pairs with a larger optimal rotation angle.

## Running out of steam

The theory behind using rotations rather than offsets is that the offset pushes the vector off the spherical geometry of the vector embedding on a tangent that might not make sense for the initial vector... If a rotation makes sense, the plane it rotates on should be similar for all pairs.

... so are the planes of rotation similar?


```python
def compute_plane_similarity(A, B, C, D):
    def orthonormal_basis(vec1, vec2):
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_proj = vec2 - np.dot(vec2, vec1_norm) * vec1_norm
        vec2_norm = vec2_proj / np.linalg.norm(vec2_proj)
        return np.stack([vec1_norm, vec2_norm], axis=1)
    
    plane1 = orthonormal_basis(A, B)
    plane2 = orthonormal_basis(C, D)
    
    M = np.dot(plane1.T, plane2)
    _, singular_values, _ = np.linalg.svd(M)
    
    angles = np.arccos(np.clip(singular_values, -1.0, 1.0))

    total = 1
    for angle in angles:
        total *= np.cos(angle)
    return total

pairs = [
    (man, woman),
    (king, queen),
    (prince, princess),
    (son, daughter),
    (actor, actress),
    (steward, stewardess),
]
        
res = np.zeros(shape=(len(gender_vectors), len(gender_vectors)))

for r in range(len(gender_vectors)):
    for c in range(len(gender_vectors)):
        target_a, source_a = pairs[r]
        target_b, source_b = pairs[c]
        res[r, c] = compute_plane_similarity(source_a, target_a, source_b, target_b)
pd.DataFrame(res)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.120395</td>
      <td>0.106358</td>
      <td>0.149667</td>
      <td>0.162984</td>
      <td>0.082681</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.120395</td>
      <td>1.000000</td>
      <td>0.555729</td>
      <td>0.059285</td>
      <td>0.092489</td>
      <td>0.097893</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.106358</td>
      <td>0.555729</td>
      <td>1.000000</td>
      <td>0.060170</td>
      <td>0.077986</td>
      <td>0.099463</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.149667</td>
      <td>0.059285</td>
      <td>0.060170</td>
      <td>1.000000</td>
      <td>0.072818</td>
      <td>0.037598</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.162984</td>
      <td>0.092489</td>
      <td>0.077986</td>
      <td>0.072818</td>
      <td>1.000000</td>
      <td>0.091440</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.082681</td>
      <td>0.097893</td>
      <td>0.099463</td>
      <td>0.037598</td>
      <td>0.091440</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Nope, they are not aligned at all... It would have been great to have checked this first.

## Does rotation work for other concepts?


```python
for target, source, pairs in [
    (
        "parent",
        "child",
        [
            ("father", "son"),
            ("mother", "daughter"),
            ("dog", "puppy"),
        ],
    ),
    (
        "Group",
        "Individual",
        [
            ("a people", "a person"),
            ("nation", "citizen"),
        ],
    ),
    (
        "Positive",
        "Negative",
        [
            ("happy", "sad"),
            ("love", "hate"),
            ("success", "failure"),
            ("hot", "cold"),
            ("light", "dark"),
        ],
    ),
    (
        "Plural noun",
        "Singular noun",
        [
            ("people", "person"),
            ("children", "child"),
            ("mice", "mouse"),
        ],
    ),
]:

    target_v, source_v = get_embedding_openai([target, source])
    pair_vecs = [get_embedding_openai(pair) for pair in pairs]
    rotation, angle = compute_nd_rotation_matrix(source_v, target_v)
    offset = target_v - source_v
    print(f"Processing {source} to {target}")
    for t, s in pair_vecs:
        display(
            style_results(
                compute_results(target=t, source=s, rotation=rotation, offset=offset)
            )
        )
        print()
    print()
    print()

```

    Processing child to parent



<style type="text/css">
#T_ce29b_row1_col0, #T_ce29b_row1_col1 {
  font-weight: bold;
}
</style>
<table id="T_ce29b">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_ce29b_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_ce29b_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_ce29b_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_ce29b_row0_col0" class="data row0 col0" >0.468058</td>
      <td id="T_ce29b_row0_col1" class="data row0 col1" >31.761523</td>
    </tr>
    <tr>
      <th id="T_ce29b_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_ce29b_row1_col0" class="data row1 col0" >0.543565</td>
      <td id="T_ce29b_row1_col1" class="data row1 col1" >29.534604</td>
    </tr>
    <tr>
      <th id="T_ce29b_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_ce29b_row2_col0" class="data row2 col0" >0.443090</td>
      <td id="T_ce29b_row2_col1" class="data row2 col1" >32.606579</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_66635_row0_col0, #T_66635_row0_col1 {
  font-weight: bold;
}
</style>
<table id="T_66635">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_66635_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_66635_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_66635_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_66635_row0_col0" class="data row0 col0" >0.669960</td>
      <td id="T_66635_row0_col1" class="data row0 col1" >25.355998</td>
    </tr>
    <tr>
      <th id="T_66635_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_66635_row1_col0" class="data row1 col0" >0.648269</td>
      <td id="T_66635_row1_col1" class="data row1 col1" >25.751596</td>
    </tr>
    <tr>
      <th id="T_66635_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_66635_row2_col0" class="data row2 col0" >0.644994</td>
      <td id="T_66635_row2_col1" class="data row2 col1" >25.854934</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_54a53_row0_col0, #T_54a53_row0_col1 {
  font-weight: bold;
}
</style>
<table id="T_54a53">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_54a53_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_54a53_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_54a53_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_54a53_row0_col0" class="data row0 col0" >0.559153</td>
      <td id="T_54a53_row0_col1" class="data row0 col1" >29.110067</td>
    </tr>
    <tr>
      <th id="T_54a53_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_54a53_row1_col0" class="data row1 col0" >0.353922</td>
      <td id="T_54a53_row1_col1" class="data row1 col1" >35.056743</td>
    </tr>
    <tr>
      <th id="T_54a53_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_54a53_row2_col0" class="data row2 col0" >0.464196</td>
      <td id="T_54a53_row2_col1" class="data row2 col1" >31.915750</td>
    </tr>
  </tbody>
</table>



    
    
    
    Processing Individual to Group



<style type="text/css">
#T_f572d_row0_col0, #T_f572d_row0_col1 {
  font-weight: bold;
}
</style>
<table id="T_f572d">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f572d_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_f572d_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f572d_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_f572d_row0_col0" class="data row0 col0" >0.687920</td>
      <td id="T_f572d_row0_col1" class="data row0 col1" >24.467766</td>
    </tr>
    <tr>
      <th id="T_f572d_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_f572d_row1_col0" class="data row1 col0" >0.549779</td>
      <td id="T_f572d_row1_col1" class="data row1 col1" >29.304673</td>
    </tr>
    <tr>
      <th id="T_f572d_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_f572d_row2_col0" class="data row2 col0" >0.648763</td>
      <td id="T_f572d_row2_col1" class="data row2 col1" >26.148103</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_75667_row0_col0, #T_75667_row0_col1 {
  font-weight: bold;
}
</style>
<table id="T_75667">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_75667_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_75667_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_75667_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_75667_row0_col0" class="data row0 col0" >0.524459</td>
      <td id="T_75667_row0_col1" class="data row0 col1" >30.635595</td>
    </tr>
    <tr>
      <th id="T_75667_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_75667_row1_col0" class="data row1 col0" >0.316418</td>
      <td id="T_75667_row1_col1" class="data row1 col1" >36.517791</td>
    </tr>
    <tr>
      <th id="T_75667_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_75667_row2_col0" class="data row2 col0" >0.442958</td>
      <td id="T_75667_row2_col1" class="data row2 col1" >33.354509</td>
    </tr>
  </tbody>
</table>



    
    
    
    Processing Negative to Positive



<style type="text/css">
#T_bd5ee_row1_col0, #T_bd5ee_row1_col1 {
  font-weight: bold;
}
</style>
<table id="T_bd5ee">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_bd5ee_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_bd5ee_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_bd5ee_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_bd5ee_row0_col0" class="data row0 col0" >0.609937</td>
      <td id="T_bd5ee_row0_col1" class="data row0 col1" >27.556616</td>
    </tr>
    <tr>
      <th id="T_bd5ee_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_bd5ee_row1_col0" class="data row1 col0" >0.660815</td>
      <td id="T_bd5ee_row1_col1" class="data row1 col1" >25.557532</td>
    </tr>
    <tr>
      <th id="T_bd5ee_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_bd5ee_row2_col0" class="data row2 col0" >0.657954</td>
      <td id="T_bd5ee_row2_col1" class="data row2 col1" >25.656030</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_20e97_row2_col0, #T_20e97_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_20e97">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_20e97_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_20e97_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_20e97_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_20e97_row0_col0" class="data row0 col0" >0.373151</td>
      <td id="T_20e97_row0_col1" class="data row0 col1" >34.858771</td>
    </tr>
    <tr>
      <th id="T_20e97_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_20e97_row1_col0" class="data row1 col0" >0.388448</td>
      <td id="T_20e97_row1_col1" class="data row1 col1" >33.971496</td>
    </tr>
    <tr>
      <th id="T_20e97_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_20e97_row2_col0" class="data row2 col0" >0.402705</td>
      <td id="T_20e97_row2_col1" class="data row2 col1" >33.769119</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_05194_row2_col0, #T_05194_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_05194">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_05194_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_05194_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_05194_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_05194_row0_col0" class="data row0 col0" >0.597542</td>
      <td id="T_05194_row0_col1" class="data row0 col1" >27.482831</td>
    </tr>
    <tr>
      <th id="T_05194_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_05194_row1_col0" class="data row1 col0" >0.618126</td>
      <td id="T_05194_row1_col1" class="data row1 col1" >27.209885</td>
    </tr>
    <tr>
      <th id="T_05194_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_05194_row2_col0" class="data row2 col0" >0.632441</td>
      <td id="T_05194_row2_col1" class="data row2 col1" >26.262374</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_ee667_row2_col0, #T_ee667_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_ee667">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_ee667_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_ee667_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_ee667_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_ee667_row0_col0" class="data row0 col0" >0.554468</td>
      <td id="T_ee667_row0_col1" class="data row0 col1" >29.435530</td>
    </tr>
    <tr>
      <th id="T_ee667_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_ee667_row1_col0" class="data row1 col0" >0.514706</td>
      <td id="T_ee667_row1_col1" class="data row1 col1" >30.383623</td>
    </tr>
    <tr>
      <th id="T_ee667_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_ee667_row2_col0" class="data row2 col0" >0.565492</td>
      <td id="T_ee667_row2_col1" class="data row2 col1" >28.948688</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_5e840_row2_col0, #T_5e840_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_5e840">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_5e840_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_5e840_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_5e840_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_5e840_row0_col0" class="data row0 col0" >0.448411</td>
      <td id="T_5e840_row0_col1" class="data row0 col1" >32.322260</td>
    </tr>
    <tr>
      <th id="T_5e840_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_5e840_row1_col0" class="data row1 col0" >0.458061</td>
      <td id="T_5e840_row1_col1" class="data row1 col1" >32.072349</td>
    </tr>
    <tr>
      <th id="T_5e840_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_5e840_row2_col0" class="data row2 col0" >0.480533</td>
      <td id="T_5e840_row2_col1" class="data row2 col1" >31.395405</td>
    </tr>
  </tbody>
</table>



    
    
    
    Processing Singular noun to Plural noun



<style type="text/css">
#T_9be60_row0_col0, #T_9be60_row0_col1 {
  font-weight: bold;
}
</style>
<table id="T_9be60">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_9be60_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_9be60_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_9be60_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_9be60_row0_col0" class="data row0 col0" >0.847886</td>
      <td id="T_9be60_row0_col1" class="data row0 col1" >17.133105</td>
    </tr>
    <tr>
      <th id="T_9be60_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_9be60_row1_col0" class="data row1 col0" >0.720367</td>
      <td id="T_9be60_row1_col1" class="data row1 col1" >23.143023</td>
    </tr>
    <tr>
      <th id="T_9be60_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_9be60_row2_col0" class="data row2 col0" >0.839824</td>
      <td id="T_9be60_row2_col1" class="data row2 col1" >17.431185</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_a62ad_row0_col0, #T_a62ad_row0_col1 {
  font-weight: bold;
}
</style>
<table id="T_a62ad">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_a62ad_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_a62ad_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a62ad_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_a62ad_row0_col0" class="data row0 col0" >0.832203</td>
      <td id="T_a62ad_row0_col1" class="data row0 col1" >18.139276</td>
    </tr>
    <tr>
      <th id="T_a62ad_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_a62ad_row1_col0" class="data row1 col0" >0.671063</td>
      <td id="T_a62ad_row1_col1" class="data row1 col1" >25.081717</td>
    </tr>
    <tr>
      <th id="T_a62ad_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_a62ad_row2_col0" class="data row2 col0" >0.825308</td>
      <td id="T_a62ad_row2_col1" class="data row2 col1" >18.384035</td>
    </tr>
  </tbody>
</table>



    



<style type="text/css">
#T_b028b_row0_col0, #T_b028b_row2_col1 {
  font-weight: bold;
}
</style>
<table id="T_b028b">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b028b_level0_col0" class="col_heading level0 col0" >cosine_similarity</th>
      <th id="T_b028b_level0_col1" class="col_heading level0 col1" >euclidean_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b028b_level0_row0" class="row_heading level0 row0" >Original Vector</th>
      <td id="T_b028b_row0_col0" class="data row0 col0" >0.561507</td>
      <td id="T_b028b_row0_col1" class="data row0 col1" >28.809626</td>
    </tr>
    <tr>
      <th id="T_b028b_level0_row1" class="row_heading level0 row1" >Summed Vector</th>
      <td id="T_b028b_row1_col0" class="data row1 col0" >0.453895</td>
      <td id="T_b028b_row1_col1" class="data row1 col1" >32.022517</td>
    </tr>
    <tr>
      <th id="T_b028b_level0_row2" class="row_heading level0 row2" >Rotated Vector</th>
      <td id="T_b028b_row2_col0" class="data row2 col0" >0.559407</td>
      <td id="T_b028b_row2_col1" class="data row2 col1" >28.808987</td>
    </tr>
  </tbody>
</table>



    
    
    


# Useful maybe?

From applying this rotation strategy to other concepts, we can see that for all but "Positive/Negative", these rotations hurt more than they help (as do the offsets).

Is it possible that positive/negative and gender are so baked into language that they have a special place in the embedding? It is more likely that I'm simply reading the tea leaves. 

While this exploration was unsuccessful, I'm glad I did it. It's good to know that the vector addition interpretations for word embeddings don't necessarily apply to sentence embeddings in a way that is immediately interpretable.

## Limitations
1. We don't know if these learnings apply to all models. From brief testing, the rotation for gender falls apart when using "all-MiniLM-L6-v2" on HuggingFace.
2. Perhaps averaged vectors could lead to substantially better results. From brief testing, this resulted in fewer cases where the vector was further away from the target but wasn't a silver bullet.