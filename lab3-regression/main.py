import prepare_data as prep_d
import one_factor_model as ofm
import multifactor_model as mfm


def main() -> None:
  """Run lab3 demo: create data, run one-factor and multi-factor regressions."""
  # create base dataset and run one-factor analysis + visualization
  prep_d.create_data()
  ofm.model_analysis()
  try:
    ofm.visualize_results()
  except RuntimeError as exc:
    # plotting libs missing — continue without plotting
    print(f"Skipping visualization: {exc}")

  # create dataset with AvgVisitors and run multi-factor regression
  prep_d.create_data(True)
  mfm.run_regression()


if __name__ == "__main__":
  main()
