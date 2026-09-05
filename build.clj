(ns build
  (:require [clojure.java.io :as io]
            [clojure.java.shell :as shell])
  (:import [java.util.jar JarOutputStream JarEntry]))

(def version "0.2.0-SNAPSHOT")
(def artifact (str "target/hnsw-clj-" version ".jar"))

(defn -main [& _]
  (let [{:keys [exit err]} (shell/sh "bash" "scripts/compile_vector_kernel.sh")]
    (when-not (zero? exit) (throw (ex-info "Java compilation failed" {:stderr err}))))
  (with-open [output (JarOutputStream. (io/output-stream artifact))]
    (doseq [root ["src" "target/classes"]
            file (sort-by str (file-seq (io/file root)))
            :when (and (.isFile file) (re-find #"\.(clj|class)$" (.getName file)))]
      (let [name (str (.relativize (.toPath (io/file root)) (.toPath file)))]
        (.putNextEntry output (doto (JarEntry. name) (.setTime 0)))
        (io/copy file output)
        (.closeEntry output)))
    (doseq [[name content]
            [["META-INF/maven/io.github.damesek/hnsw-clj/pom.xml"
              (str "<project xmlns=\"http://maven.apache.org/POM/4.0.0\"><modelVersion>4.0.0</modelVersion>"
                   "<groupId>io.github.damesek</groupId><artifactId>hnsw-clj</artifactId><version>"
                   version "</version><name>hnsw-clj</name>"
                   "<description>JVM-native HNSW approximate nearest-neighbor library</description>"
                   "<url>https://github.com/damesek/hnsw-clj</url><dependencies><dependency>"
                   "<groupId>org.clojure</groupId><artifactId>clojure</artifactId>"
                   "<version>1.11.1</version></dependency></dependencies></project>")]
             ["META-INF/maven/io.github.damesek/hnsw-clj/pom.properties"
              (str "groupId=io.github.damesek\nartifactId=hnsw-clj\nversion=" version "\n")]]]
      (.putNextEntry output (doto (JarEntry. name) (.setTime 0)))
      (.write output (.getBytes ^String content "UTF-8"))
      (.closeEntry output)))
  (println artifact))
