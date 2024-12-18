# How to Find logs

By default, GraphScope is running in a silent mode following the convention of Python applications. 
To enable verbose logging, turn it on by this command after importing `graphscope`.

```python
>>> import graphscope
>>> graphscope.set_option(show_log=True)
```

## Find logs in k8s
If you are running GraphScope in k8s, you can use [kubectl describe](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands#describe) and [kubectl logs](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands#logs) to check the log/status of the cluster.

```shell
# list graphscope pods
$ kubectl get pod
NAME                                                    READY   STATUS    RESTARTS      AGE
coordinator-syoove-79b44f7b58-ctmqb                     1/1     Running   0             10m
gs-engine-syoove-0                                      5/5     Running   0             9m53s
gs-engine-syoove-1                                      5/5     Running   0             9m46s
gs-interactive-frontend-syoove-6dd67c65fc-gn4mb         1/1     Running   0             9m53s

# describe the status of pod
$ kubectl describe pod <pod_name>

# print the logs of pod
$ kubectl logs -f <pod_name>
```

### Find logs for GraphScope Analytical Engine (GAE)

Within the k8s environment, all logs for the analytical engine (GAE) are consolidated into the coordinator pod. Therefore, you can view GAE logs by using the command:

```shell
$ kubectl logs -f coordinator-syoove-79b44f7b58-ctmqb
```

Additionally, if you can access into the engine pod, you can find all logs in `/tmp/grape_engine.INFO` file.

```shell
$ kubectl exec -it gs-engine-syoove-0 -c engine -- /bin/bash
$ cat /tmp/grape_engine.INFO
```

### Find logs for Graph Interactive Engine (GIE)

When using the Graph Interactive Engine (GIE):

```python
>>> # g is a property graph in session
>>> interactive = sess.gremlin(g)
```

It it often necessary to view the `frontend` and `executor` logs. The `frontend` logs contain the logical query plans generated by the compiler, while the `executor` is the actual execution engine.

**Frontend:** You can find the frontend logs in the `/var/log/graphscope` directory within the `frontend` pod, by following these steps:

```shell
$ kubectl exec -it gs-interactive-frontend-syoove-6dd67c65fc-gn4mb -- /bin/bash
$ cd /var/log/graphscope/15334625083466732 && tail -f frontend.log
```

The number `15334625083466732` mentioned above is the graph ID behind the GIE. If you have created multiple GIE instances, you can find the corresponding graph ID using `g.vineyard_id`.


```python
>>> # g is a property graph in session
>>> interactive = sess.gremlin(g)
>>> g.vineyard
15334625083466732
```

**Executor:** Similarly, you can find the executor logs in the `/var/log/graphscope` directory within the `executor` container of the engine pod.

```shell
$ kubectl exec -it gs-engine-syoove-0 -c executor -- /bin/bash
$ cd /var/log/graphscope/15334625083466732 && tail -f executor.0.log
```

### Find logs for Graph Learning Engine (GLE)

When using the Graph Learning Engine (GLE):

```python
>>> # g is a property graph in session
>>> lg = session.graphlearn(g, ...)
```

You can find the learning logs in the `/home/graphscope/graphlearn.INFO` file within the `learning` container of the engine pod:

```bash
$ kubectl exec -it gs-engine-syoove-0 -c learning -- /bin/bash
$ cat /home/graphscope/graphlearn.INFO
```

## Find logs for Groot

It is common to find the logs of Frontend and Store roles. When debugging, it is often necessary to find the logs of Coordinator as well. The logs of Frontend include the logs of the Compiler that generates the logical query plan, while the logs of Store include the logs of the query engine execution. You can find the logs of each Pod using the [kubectl command](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands) `kubectl logs ${POD_NAME}`. For example,

```bash
$ kubectl get pod
NAME                                              READY   STATUS    RESTARTS      AGE
demo-graphscope-store-coordinator-0               1/1     Running   0             33d
demo-graphscope-store-frontend-0                  1/1     Running   0             33d
demo-graphscope-store-store-0                     1/1     Running   0             33d
demo-graphscope-store-store-1                     1/1     Running   0             33d

# print the last 10 lines log for frontend pod
$ kubectl logs -f demo-graphscope-store-frontend-0 --tail=10
```

Additionally, if you can access into the pod, you can find all logs in `/var/log/graphscope` directory.

```bash
# frontend
$ kubectl exec -it demo-graphscope-store-frontend-0 -- /bin/bash
$ cd /var/log/graphscope && ls
graphscope-store.2023-03-11.0.log  graphscope-store.2023-03-19.0.log  graphscope-store.log ...
```