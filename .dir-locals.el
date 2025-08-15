((clojure-mode
  (eval . (let ((port (or (ignore-errors (cider--find-nrepl-port)) 7888)))
            (unless (cider-connected-p)
              (cider-connect-clj `(:host "localhost" :port ,port)))))
  (cider-repl-pop-to-buffer-on-connect . t)))
