# Proximal Ratio (PR) — Understanding Summary

**Source:** Amer, Ravana & Habeeb (2025), "Effective k-nearest neighbor models for data classification enhancement", *Journal of Big Data*, 12:86.

## What PR Does

The Proximal Ratio is a per-training-point score in [0, 1] that measures how consistent a point is with its local same-class neighborhood. It explicitly identifies three kinds of training points:

- **PR = 1 (clean/interior):** All nearby neighbors within the class radius belong to the same class. The point sits firmly inside its own class region (e.g., Points A and C in Fig. 2).
- **0 < PR < 1 (overlap/boundary):** Some neighbors within the radius belong to a different class. The point lies in a class-overlap zone (e.g., Point B in Fig. 2, PR = 2/3).
- **PR = 0 (outlier/noise):** None of the radius-local neighbors share the point's label. The point is entirely surrounded by foreign-class points (e.g., Point D in Fig. 2, PR = 0/3 = 0).

## Computation Steps

### Step 1: Class-wise Radius (Eq. 1 / Eq. 4)

For each class c with N_c points, compute:

```
R_c = (sum of ALL pairwise Euclidean distances between points in class c) / N_c
```

This is **not** the conventional mean pairwise distance (which would divide by N_c*(N_c-1) or N_c^2). The paper divides the full double-sum by N_c only, making R_c approximately equal to the average *total* distance from one point to every other point in its class. This yields a much larger radius than the conventional mean, which is intentional — it defines a generous hyper-sphere around each point.

Eq. 4 restates this equivalently as: for each point i in class c, let dist_i = sum of distances from i to all other points in class c. Then R_c = (sum of all dist_i) / N_c. Since each pair (i,j) appears twice in the double-sum (as dist(i,j) and dist(j,i)), Eq. 1 and Eq. 4 are identical.

### Step 2: Proximal Ratio per Point (Eq. 2 / Eq. 3)

For training point t with label y_t:

1. Find the k nearest neighbors of t from the **full** training set (all classes), excluding t itself.
2. Among those k neighbors, keep only those whose distance to t is <= R_{y_t} (the class radius of t's own class). Call this filtered set S.
3. Count val = number of points in S that share t's label.
4. Compute PR(t) = val / |S|. If |S| = 0 (no neighbors within the radius), this is an edge case — the paper does not explicitly define it, so we default to PR = 1.0 (an isolated point with no local evidence of overlap should not be penalized).

## Figure 2 Worked Example (k = 3)

The figure shows two classes: Class 1 (red circles) and Class 2 (blue triangles), with four labeled points A, B, C, D.

- **Point A (Class 1):** 4 points lie within its class radius, but k=3 so we consider the 3 nearest. All 3 are Class 1. PR = 3/3 = **1.0**.
- **Point B (Class 1):** 5 points within radius, consider 3 nearest. 2 are Class 1, 1 is Class 2. PR = 2/3 = **0.667**. Point B is in the overlap zone.
- **Point C (Class 2):** Only 1 neighbor falls within the class radius, and it is Class 2. PR = 1/1 = **1.0**. Despite having few radius-local neighbors, the one present is same-class.
- **Point D (Class 2):** 3 neighbors within radius, but 0 are Class 2. PR = 0/3 = **0.0**. Point D is an outlier — embedded in the Class 1 region.

## How PR Is Used in the Paper (PRkNN model)

In the paper's first model (PRkNN), PR scores computed on training data are used at **test time** as neighbor weights: each training-point neighbor's vote is weighted by PR_i / distance_i (Eq. 5), then class-wise weights are averaged (Eq. 6), and the class with the highest weight is predicted (Eq. 7). This is a weighted kNN scheme.

**Our adaptation (PW-NB)** differs: instead of using PR to weight kNN votes at test time, we use PR as **sample weights during Naive Bayes training**. Outliers (PR=0) contribute nothing to likelihood estimation; boundary points contribute partially; clean points contribute fully. This makes NB's density estimates more robust to overlap and noise without changing the NB prediction mechanism itself.
