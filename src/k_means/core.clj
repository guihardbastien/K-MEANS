(ns k-means.core)

(defn sum [list] (reduce + 0 list))

;; map containing configs -> 2 clusters
(def configuration {:k-clusters    2
                    :max-epochs    1000
                    :threshold     0.0001
                    :bounds        [[0 15] [0 15]]          ;; 2 dim
                    :current-epoch 0})

;; points list of 2 coordinates
(def points [[0 0] [0 1] [1 0] [1 1]
             [10 10] [10 11] [11 10] [11 11]])

;; map containing configs -> 2 clusters
(def configuration-3d {:k-clusters    2
                       :max-epochs    1000
                       :threshold     0.0001
                       :bounds        [[0 15] [0 15] [0 15]] ;; 2 dim
                       :current-epoch 0})

;; points list of 2 coordinates
(def points-3d [[0 0 0] [0 1 0] [1 0 1] [1 1 0]
                [10 10 10] [10 11 10] [11 10 11] [11 11 11]])

;; bounds should be an array of arrays eg [[0 10][0 10][0 10]]
(defn generate-centroid [bounds]
  (mapv (fn [bound] (rand-nth (range (first bound) (second bound)))) bounds))

(generate-centroid [[0 15] [0 15]])

(defn initial-state [config]
  (let [centroids (mapv (fn [elt] (generate-centroid (get config :bounds))) (range (get config :k-clusters)))]
    (assoc config :groups (vec (repeat (count centroids) #{}))
                  :centroids centroids)))

(defn transpose-matrix [matrix]
  (apply mapv vector matrix))

(defn euclidean-distance [point-1 point-2]
  (Math/sqrt
    (sum
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
(main configuration-3d points-3d)