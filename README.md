
# Data format
In the end, our dataset requres three datasets:
* `accounts.csv`, recording the `account_id`, `internal`, and `label` for fraud or not if the account is `internal`. We leave labels NaN for external accounts.
* `trasactions.csv`, recording the transaction from account 1 (`sender`) to account 2 (`receiver`), with transaction type `txn_type`: 0 (internal transactions), 1 (internal->external), and 2(external->internal).
* `features.csv`, recording the node representations for internal accounts. The node features for the external accounts are all set to 0?

# Elliptic dataset
The [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) dataset description:
>The graph is made of 203,769 nodes and 234,355 edges. Two percent (4,545) of the nodes are labelled class1 (illicit). Twenty-one percent (42,019) are labelled class2 (licit). The remaining transactions are not labelled with regard to licit versus illicit.

# DGraph-Fin dataset
The [DGraph-Fin](https://dgraph.xinye.com/dataset):
> The graph is made of 3,700,550 nodes and 4,300,999 edges. Background accounts (2,474,949 in total) are naturally considered as the external accounts, taking up 67% percents. The rest 1,225,601 accounts are internal accounts. 1% of accounts in total are malicious accounts.