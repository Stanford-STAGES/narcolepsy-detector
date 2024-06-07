from narcolepsy_detector.utils.argparser import eval_arguments
from narcolepsy_detector.evaluate.evaluate_single_model import evaluate_single_model


if __name__ == "__main__":

    args = eval_arguments()

    evaluate_single_model(
        data=args.data_dir,
        experiment=args.experiment_dir,
        savedir_output=args.savedir_output,
        data_master=args.data_master,
    )
