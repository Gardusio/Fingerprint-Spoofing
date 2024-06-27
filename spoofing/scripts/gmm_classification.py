from lib.models.gmm_classifier import *
from lib.evaluation.bayes_evaluator import BinaryBayesEvaluator
from lib.evaluation.application import Application


def run_gmm_classification(ds):
    print("-" * 80)
    print("\n Classifying with GMM on spoofing dataset...")

    main_app = Application(0.1, 1, 1)

    x_train, y_train, x_val, y_val = ds.split_ds_2to1()

    for cov_type in ["full", "diagonal", "tied"]:
        model_name = f"GMMClassifier-{cov_type}"
        gmm_classifier = GMMClassifier(cov_type=cov_type, name=model_name)

        for components in [1, 2, 4, 8, 16, 32]:
            gmm0 = gmm_classifier.fit(
                x_train[:, y_train == 0], components, verbose=False, psi_eig=0.01
            )
            gmm1 = gmm_classifier.fit(
                x_train[:, y_train == 1], components, verbose=False, psi_eig=0.01
            )

            llrs, predictions = gmm_classifier.classify(x_val, gmm1, gmm0)

            evaluator = BinaryBayesEvaluator(
                application=main_app,
                evaluation_labels=y_val,
                llrs=llrs,
                opt_predictions=predictions,
                model_name=f"{model_name}-COMPONENTS-{components}",
            )

            # print ('minDCF - pT = 0.5: %.4f' % bayesRisk.compute_minDCF_binary_fast(llrs, LVAL, 0.5, 1.0, 1.0))
            # print ('actDCF - pT = 0.5: %.4f' % bayesRisk.compute_actDCF_binary_fast(llrs, LVAL, 0.5, 1.0, 1.0))
            print(
                "\tComponents = %d: %.4f / %.4f"
                % (
                    components,
                    evaluator.compute_mindcf(),
                    evaluator.compute_dcf(),
                )
            )
        print()

    print("-" * 80)
