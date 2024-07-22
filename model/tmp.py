def recon_upsample(embed, labels, idx_train, transactions, target_portion=1.0):
    embed = embed.cpu()  # Ensure everything is on CPU for debugging
    labels = labels.cpu()
    idx_train = idx_train.cpu()
    transactions = transactions.copy()
    portion = target_portion
    
    c_largest = 1
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    adj_new = None
    new_transactions = []
    new_node_names = []
    
    for i in range(1):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.detach().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(1)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            # Generate unique node names for new nodes
            for k in range(new_embed.size(0)):
                new_node_name = f'new_node_{idx_new[k].item()}'
                new_node_names.append(new_node_name)

                # Generate new transactions for the new nodes
                parent_index = chosen[k].item()
                new_index = idx_new[k].item()

                # Copy transactions involving the parent node to the new node
                parent_transactions = transactions[(transactions['sender'] == parent_index) | (transactions['receiver'] == parent_index)]
                new_transactions_sender = parent_transactions.copy()
                new_transactions_sender['sender'].replace(parent_index, new_index, inplace=True)
                new_transactions_receiver = parent_transactions.copy()
                new_transactions_receiver['receiver'].replace(parent_index, new_index, inplace=True)

                # Append new transactions to the list of new transactions
                new_transactions.append(new_transactions_sender)
                new_transactions.append(new_transactions_receiver)

    # Combine all new transactions into a single DataFrame
    new_transactions = pd.concat(new_transactions, ignore_index=True)

    return embed, labels, idx_train, new_transactions, new_node_names