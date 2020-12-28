# Introduction to clojure - k-means from scratch 

## Introduction

### K-means intuition

![k-means](https://i.imgur.com/k4XcapI.gif)

Before getting started with clojure, let's have a look at the algorithm we're trying to build.

According to wikipedia, the K-means algorithm is meant to partition *n* observations into *k* clusters in which each observation belongs to the **cluster** with the nearest **mean** (cluster centers or cluster centroid)

To put it differently, the k-means algorithm aims at finding the best spot for each centroid to sit.



__How come we do that ?__

In theory it's not that hard. Here are the steps to get there: 

- First we'll need to randomly initialize *k* centroids in our vector space

- Then, we'll need to compute the distance that separates each point from each centroid

- Each point will be associated to it's closest centroid

- At this point we have a first *-rather random-* classification.

- We can now update our centroids by computing the mean point of each cluster 
- Repeat (but obviously skip the first step) until convergence



**If you just want to play around you'll find the full code at the end of the blog post.**



### Clojure basics

There are a few prerequisite for this article :

- You should have clojure installed on your machine. If not, you can refer to the official guide https://clojure.org/guides/getting_started
- We'll be using IntelliJ IDEA with the **[Cursive](https://cursive-ide.com/)** extension (download it in *Preference -> Plugins* then type cursive in the search bar)

For us to implement our algorithm, we'll need to understand a thing or two about clojure.

*Shaunlebron* did a good job on explaining basic clojure syntax **I strongly suggest you give it a look [here](https://github.com/shaunlebron/ClojureScript-Syntax-in-15-minutes)**

We'll get into more details during the implementation.



### Terminology used in this article

`centroid`: central point of a point cluster

`cluster`: aggregation of data

## Prelude: Set up

First of all we'll define a global variable using the `def` keyword that will hold an **associative array** of our configuration.

```clojure
(def configuration {:k-clusters    2							
                    :max-epochs    1000						
                    :threshold     0.0001					
                    :bounds        [[0 15] [0 15]]  ;; 2 dimensions
                    :current-epoch 0})						
```



Let's define what's in this associative array:

- `:k-clusters`: The amount of clusters we want our data to be classified in

- `:max-epochs`: The maximum loop we're allow to make before convergence

- `:threshold`: The threshold used to define wether our model converged or not

- `:bounds`: Bounds is a vector of m-dimensional-vectors that defines the vector space of our dataset

- `:current-epoch`: As our model converges, it will return the current loop we're in.

  

Also, let's build a small data set that we'll use as a validation dataset

```clojure
(def points [[0 0] [0 1] [1 0] [1 1]
             [10 10] [10 11] [11 10] [11 11]])
```

By now you should be able to say *"We defined a global variable called `points` that is a vector of 2 dimensional vectors"*



## Step one: Randomly initialize centroids

Our aim here is: given our vector space (`:bounds` in our previous associative array) we'd like to generate *n* random points that will be our initial centroids.

**Do we have special tools for this ?**

Yes ! We'll use `mapv`, `rand-nth`and `range`

Before defining `mapv` let's define `map`

- The basic `map` usage allows you to pass two arguments to the map function : a function and a sequence fo elements.

Example: 

```clojure
(map (fn [elt] (* elt 2)) [1 2 3])
;; the output will be (2 4 6)
```

In this example map applies a lambda function `(fn [elt] (* elt 2))` and applies it to every element of the array`[1 2 3]`. Then it returns a new list of results.

> Keep in mind that in clojure, `map` is lazy by design.
>
> Also, you might be thinking *"returning a new copy of each data everytime we want to do something with it must be incredibly inneficient !"* - well just like in java you usually return an updated `view` of the updated data. We'll discuss this more in another topic but for now you can assume this is *copy-paste* on steroids.



Getting back to `mapv` well basically it will return a vector instead of a list.

```clojure
(mapv (fn [elt] (* elt 2)) [1 2 3])
;; the output will be [2 4 6]
```



`rand-nth`and `range` on the other hand are pretty straight forward :

```clojure
(range 10 15)
;; outputs -> (10 11 12 13 14) 
(rand-nth (range 10 15))
;; pick a random number in the range. outputs -> 13 (for instance)
```



So how do we generate our our random centroid ?

First we'll define a function that takes some **bounds for argument** remember bounds are defined as a vector of vectors in our configuration.

Then we can use the `mapv` function to iterate through the bounds of each dimension and return a centroid vector !

```clojure
(defn generate-centroid [bounds]
  (mapv (fn [bound] (rand-nth (range (first bound) (second bound)))) bounds))

;;try it out like so 
(generate-centroid  [[0 15] [0 15]] )
```

## Step Two: Let's build our application state using our `configuration`



Now that we can initialize random centroids, let's build an initial state that we'll make converge throughout the algorithm.

For this, let us first introduce `let`.

`let` allow you to define scoped constants that you might want to use in your return value.

For instance let's make a function that returns a vector of computed values:

```clojure
(defn fun [x y]
  (let [x-times-two (* 2 x)
        y-times-two (* 2 y)]
    [x-times-two y-times-two]))

(fun 1 2) ;; output [2 4]
```

Values like `x-times-two (* 2 x)` are called local bindings 

Let's now introduce clojure **sets**. 

Sets are a list of unique elements. In clojure they can be defined like so : `#{:bar 3.14 "hello"}`

The `assoc` operation allow you to add a new key-value pair in an associative array.

`vec `can turn a list into a vector and `repeat `returns a list of repeated element. For instance `(vec (repeat 2 "hello")) `will return `["hello" "hello"]` 



Here is how we build our initial state:

```clojure
(defn initial-state [config]
  (let [centroids (mapv (fn [elt] (generate-centroid (get config :bounds))) (range (get config :k-clusters)))]
    (assoc config :groups (vec (repeat (count centroids) #{}))
                  :centroids centroids)))
```

Output:

```clojure
=>
{:k-clusters 2,
 :max-epochs 1000,
 :threshold 1.0E-4,
 :bounds [[0 15] [0 15]],
 :current-epoch 0,
 :groups [#{} #{}], 						;; Vector of repeated empty sets
 :centroids [[3 11] [14 11]]}		;; Random centroids
```



## Step Three: Euclidean distances & point classification

Given two points **A(X1,Y1)** and **B(X2,Y2)** the euclidian distance between these two points can be computed like so :

**`√(X1-X2)²(Y1-Y2)²`**

In clojure, Math basic function can be used like this `(Math/sqrt 4)`

Now let's build a function that compute the euclidean distance between 2 **2D points**

```clojure
(defn euclidean-distance [[x1 y1] [x2 y2]]
  (Math/sqrt (+ (Math/pow (- x1 x2) 2)
                (Math/pow (- y1 y2) 2)))) 
```

We used destructuration in the previous example by providing arguments in vectors `[x1 y1] [x2 y2]`. Indeed our euclidean distance can now take two two-dimensional vectors but also we can access every coordinate directly.

However, we might want to scale our program up by using points with more than 2 dimensions



A hack to subtract every "Xs" of our points is to view our points as a transposed matrix and then reduce every vector.

For instance `[[x1 y1] [x2 y2]]` will become `[[x1 x2] [y1 y2]]` 

We can then use a `mapv - reduce` combo to return a vector differences !



So let's build a function that compute the euclidean distance between 2 **mD points**

```clojure
(defn euclidean-distance [point-1 point-2]
  (Math/sqrt
    (reduce + 0
            (mapv (fn [diff] (Math/pow diff 2))
                  (mapv (fn [nth-elts] (reduce - (first nth-elts) (rest nth-elts)))
                        (transpose-matrix [point-1 point-2]))))))
```



Remember the output of the previous step ? Now we want to create a function that can add points in each set of our state.

For this we'll use `update-in` that allow us to update an associative array given a path.

```clojure
(defn classify-point [state point]
  (let [centroids (get state :centroids) ;; we get the centroids from our state
        distances (mapv (fn [centroid] (euclidean-distance point centroid)) centroids) ;; we return the distances between a point and each centroid
        indexed-distances (map (fn [index value] [index value]) (range) distances) ;; we index the distances in order to classify the point in the correct group
        [index shortest-distance] (first (sort-by second indexed-distances))] ;; we keep the shortest "indexed distance"
    (update-in state [:groups index] (fn [set] (conj set point))))) ;; we add our point to the correct set 
```



## Step Four: Update centroids



Congratulations, you've made the first classification now we want to update our centroid with the mean point of each set.

Here is an example for a set of two dimensional points.

```clojure
(defn mean-point-for-record [points]
  (when (seq points)
    (let [nb-points (count points)
          xs-sum (reduce + 0 (map first points))
          ys-sum (reduce + 0 (map second points))]
      [(/ xs-sum nb-points) (/ ys-sum nb-points)])))
```

And now a version that can take any dimension. Notice that we use `when`to return `nil` if `points` is en empty vector

```clojure
(defn mean-point [points]
  (when (seq points)                                       
    (let [nb-points (count points)]
      (mapv
        (fn [coordinate-vec] (/ (reduce + 0 coordinate-vec) nb-points))
        (transpose-matrix points)))))
```

We can now update our state with the new centroids and we can also update our iteration count `:current-epoch`

```clojure
(defn update-centroids [state]
  (let [new-centroids (mapv mean-point (:groups state))
        new-current-epoch (inc (:current-epoch state))]
    (assoc state :centroids new-centroids :current-epoch new-current-epoch)))
```



## Step Five: Repeat until convergence

To say wether or not our model has converged, we want to either:

- Have reached our max-epoch 
- Have a centroid variation so slight that we can consider our job done

```clojure
(defn converged? [previous-state current-state]
  (or (> (get current-state :current-epoch) (get current-state :max-epochs))
      (let [distances (map euclidean-distance (:centroids current-state) (:centroids previous-state))]
        (every? (fn [distance] (< distance (:threshold current-state))) distances)))) ;; here we use every? to check is every variation is under the threshold
```

Let's wrap it up in a main function:

```clojure
(defn main [config points]
  (loop [state (initial-state config)]
    (let [with-points (reduce classify-point state points)
          with-new-centroids (update-centroids with-points)]
      (if (converged? state with-new-centroids)
        with-new-centroids
        (recur with-new-centroids)))))
```



## Full code 

```clojure
(ns test-clojure.clustering)

(def configuration {:k-clusters    2
                    :max-epochs    1000
                    :threshold     0.0001
                    :bounds        [[0 15] [0 15]]          ;; 2 dim
                    :current-epoch 0})

(def points [[0 0] [0 1] [1 0] [1 1]
             [10 10] [10 11] [11 10] [11 11]])

(defn generate-centroid [bounds]
  (mapv (fn [bound] (rand-nth (range (first bound) (second bound)))) bounds))

(defn initial-state [config]
  (let [centroids (mapv (fn [elt] (generate-centroid (get config :bounds))) (range (get config :k-clusters)))]
    (assoc config :groups (vec (repeat (count centroids) #{}))
                  :centroids centroids)))

(defn transpose-matrix [matrix]
  (apply mapv vector matrix))

(defn euclidean-distance [point-1 point-2]
  (Math/sqrt
    (reduce + 0
            (mapv (fn [diff] (Math/pow diff 2))
                  (mapv (fn [nth-elts] (reduce - (first nth-elts) (rest nth-elts)))
                        (transpose-matrix [point-1 point-2]))))))

(defn classify-point [state point]
  (let [centroids (get state :centroids)
        distances (mapv (fn [centroid] (euclidean-distance point centroid)) centroids)
        indexed-distances (map (fn [index value] [index value]) (range) distances)
        [index shortest-distance] (first (sort-by second indexed-distances))]
    (update-in state [:groups index] (fn [set] (conj set point)))))

(defn mean-point [points]
  (when (seq points)                                       
    (let [nb-points (count points)]
      (mapv
        (fn [coordinate-vec] (/ (reduce + 0 coordinate-vec) nb-points))
        (transpose-matrix points)))))

(defn update-centroids [state]
  (let [new-centroids (mapv mean-point (:groups state))
        new-current-epoch (inc (:current-epoch state))]
    (assoc state :centroids new-centroids :current-epoch new-current-epoch)))

(defn converged? [previous-state current-state]
  (or (> (get current-state :current-epoch) (get current-state :max-epochs))
      (let [distances (map euclidean-distance (:centroids current-state) (:centroids previous-state))]
        (every? (fn [distance] (< distance (:threshold current-state))) distances))))

(defn main [config points]
  (loop [state (initial-state config)]
    (let [with-points (reduce classify-point state points)
          with-new-centroids (update-centroids with-points)]
      (if (converged? state with-new-centroids)
        with-new-centroids
        (recur with-new-centroids)))))

(main configuration points) 
```

Expected output:

```clojure
{:k-clusters 2,
 :max-epochs 1000,
 :threshold 1.0E-4,
 :bounds [[0 15] [0 15]],
 :current-epoch 2,
 :groups [#{[0 0] [1 0] [1 1] [0 1]} #{[11 11] [11 10] [10 11] [10 10]}],
 :centroids [[1/2 1/2] [21/2 21/2]]}
```



## Ressources

- https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42
- https://learnxinyminutes.com/docs/edn/#:~:text=Extensible%20Data%20Notation%20(EDN)%20is,syntax%2C%20especially%20from%20untrusted%20sources.&text=The%20main%20benefit%20of%20EDN,is%20that%20it%20is%20extensible
- https://www.4clojure.com/problems
- https://clojure.org/api/cheatsheet



