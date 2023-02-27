



## Bar plots


if 0:
    data = [train_r+train_l, test_r+test_l]
    title_txt = ['Train', 'Test']
    #data = [folds_r[f_]+folds_l[f_] for f_ in range(5)]
    #title_txt = [f"fold{f_}" for f_ in range(1, 6)]
    ncond = len(data)

    for k in range(ncond):
        fpaths = data[k]
        fnames = [os.path.basename(k) for k in fpaths]
        fdirs = [os.path.dirname(k) for k in fpaths]
        subj_ids, epoch_indxs = [], []
        gender, acoustic, story = [], [], []
        for fname in fnames:
            indx_ = [pos for pos, c in enumerate(fname) if c == '_']
            subj_ids.append(int(fname[indx_[0] + 1:indx_[1]]))
            epoch_indxs.append(int(fname[indx_[2] + 1:indx_[3]]))
            gender.append(fname[indx_[6] + 1:indx_[7]])
            acoustic.append(int(fname[indx_[8] + 1:indx_[9]]))
            story.append(int(fname[indx_[10] + 1:indx_[11]]))


        gender_label = list(set(gender))
        gender_counts = [gender.count(k) for k in gender_label]
        if k == 0:
            fig, ax0 = plt.subplots(ncond, 1, sharex=True)
        ax0[k].bar(gender_label, gender_counts, width=.2)
        if k == ncond-1:
            ax0[k].set_xlabel('Attended Speaker Gender')
        ax0[k].set_ylabel('Counts')
        ax0[k].set_title(title_txt[k])

        story_label = list(set(story))
        story_counts = [story.count(k) for k in story_label]
        if k == 0:
            fig, ax1 = plt.subplots(ncond, 1, sharex=True)
        ax1[k].bar(story_label, story_counts)
        if k == ncond-1:
            ax1[k].set_xlabel('Attended Story')
        ax1[k].set_ylabel('Counts')
        ax1[k].set_title(title_txt[k])

        acoustic_label = list(set(acoustic))
        acoustic_counts = [acoustic.count(k) for k in acoustic_label]
        if k == 0:
            fig, ax2 = plt.subplots(ncond, 1, sharex=True)
        ax2[k].bar(acoustic_label, acoustic_counts)
        if k == ncond-1:
            ax2[k].set_xlabel('Acoustic condition')
        ax2[k].set_ylabel('Counts')
        ax2[k].set_title(title_txt[k])

        subj_ids_label = list(set(subj_ids))
        subj_ids_counts = [subj_ids.count(k) for k in subj_ids_label]
        if k == 0:
            fig, ax3 = plt.subplots(ncond, 1, sharex=True)
        ax3[k].bar(subj_ids_label, subj_ids_counts)
        if k == ncond-1:
            ax3[k].set_xlabel('Subject')
            ax3[k].set_xticks(list(range(1, 19)))
        ax3[k].set_ylabel('Counts')
        ax3[k].set_title(title_txt[k])

        epoch_indxs_label = list(set(epoch_indxs))
        epoch_indxs_counts = [epoch_indxs.count(k) for k in epoch_indxs_label]
        if k == 0:
            fig, ax4 = plt.subplots(ncond, 1, sharex=True)
        ax4[k].bar(epoch_indxs_label, epoch_indxs_counts)
        if k == ncond-1:
            ax4[k].set_xlabel('Epochs')
        ax4[k].set_ylabel('Counts')
        ax4[k].set_title(title_txt[k])





