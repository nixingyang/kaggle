import numpy as np
import prepare_data
import pylab
import solution_basic
from joblib import Parallel, delayed
from sklearn.cross_validation import KFold


def inspect_final_data_set_without_labels(image_index_list, seed):
    np.random.seed(seed)
    image_index_array = np.array(image_index_list)

    # Cross Validation
    fold_num = 5
    label_kfold = KFold(image_index_array.size, n_folds=fold_num, shuffle=True)

    true_records_num_list = []
    false_records_num_list = []

    for _, fold_item in enumerate(label_kfold):
        # Generate final data set
        selected_index_array = image_index_array[fold_item[0]]
        _, Y_train = solution_basic.get_record_map(selected_index_array, None)

        true_records = Y_train == 1
        true_records_num = np.sum(true_records)
        false_records_num = Y_train.size - true_records_num

        true_records_num_list.append(true_records_num)
        false_records_num_list.append(false_records_num)

    return (true_records_num_list, false_records_num_list)


def inspect_final_data_set_with_labels(image_index_list, seed):
    np.random.seed(seed)

    # Cross Validation
    fold_num = 5
    unique_label_values = np.unique(image_index_list)
    selected_label_values = np.random.choice(
        unique_label_values,
        size=np.ceil(unique_label_values.size * (fold_num - 1) / fold_num),
        replace=False,
    )

    selected_index_list = []
    for single_image_index in image_index_list:
        if single_image_index in selected_label_values:
            selected_index_list.append(single_image_index)
    selected_index_array = np.array(selected_index_list)

    _, Y_train = solution_basic.get_record_map(selected_index_array, None)

    true_records = Y_train == 1
    true_records_num = np.sum(true_records)
    false_records_num = Y_train.size - true_records_num

    return ([true_records_num], [false_records_num])


def inspect_number_of_occurrences():
    # Get image paths in the training and testing datasets
    _, training_image_index_list = prepare_data.get_image_paths_in_training_dataset()

    repeated_num = 20
    seed_array = np.random.choice(range(repeated_num), size=repeated_num, replace=False)
    records_list = Parallel(n_jobs=-1)(
        delayed(inspect_final_data_set_without_labels)(training_image_index_list, seed)
        for seed in seed_array
    )

    # repeated_num = 100
    # seed_array = np.random.choice(range(repeated_num), size=repeated_num, replace=False)
    # records_list = (Parallel(n_jobs=-1)(delayed(inspect_final_data_set_with_labels)(training_image_index_list, seed) for seed in seed_array))

    true_records_num_list = []
    false_records_num_list = []

    for single_true_records_num_list, single_false_records_num_list in records_list:
        for value in single_true_records_num_list:
            true_records_num_list.append(value)

        for value in single_false_records_num_list:
            false_records_num_list.append(value)

    for single_list in [true_records_num_list, false_records_num_list]:
        repeated_times_list = []
        min_value_list = []
        max_value_list = []
        mean_value_list = []

        for end_index in range(len(single_list)):
            current_list = single_list[0 : end_index + 1]

            repeated_times_list.append(len(current_list))
            min_value_list.append(np.min(current_list))
            max_value_list.append(np.max(current_list))
            mean_value_list.append(np.mean(current_list))

        pylab.figure()
        pylab.plot(
            repeated_times_list, min_value_list, color="yellowgreen", label="Minimum"
        )
        pylab.plot(
            repeated_times_list, max_value_list, color="lightskyblue", label="Maximum"
        )
        pylab.plot(
            repeated_times_list, mean_value_list, color="darkorange", label="Mean"
        )
        pylab.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=3,
            ncol=3,
            mode="expand",
            borderaxespad=0.0,
        )
        pylab.xlabel("Repeated Times", fontsize="large")
        pylab.ylabel("Number of Occurrences", fontsize="large")
        pylab.grid()
        pylab.show()


def inspect_number_of_images():
    # Get image paths in the training and testing datasets
    _, training_image_index_list = prepare_data.get_image_paths_in_training_dataset()

    images_number_list = []
    for current_image_index in np.unique(training_image_index_list):
        images_number_list.append(
            np.sum(np.array(training_image_index_list) == current_image_index)
        )

    # the histogram of the data with histtype="step"
    bins = np.arange(np.min(images_number_list), np.max(images_number_list) + 2) - 0.5
    _, _, patches = pylab.hist(images_number_list, bins=bins)
    pylab.setp(patches, "facecolor", "yellowgreen", "alpha", 0.75)
    pylab.xlim([bins[0], bins[-1]])
    pylab.xticks(np.arange(np.min(images_number_list), np.max(images_number_list) + 1))
    pylab.xlabel("Number of Images from the Same Person", fontsize="large")
    pylab.ylabel("Number of Occurrences", fontsize="large")
    pylab.title("Histogram of Number of Images from the Same Person")
    pylab.show()


inspect_number_of_occurrences()
