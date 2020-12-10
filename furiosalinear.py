'''
linear regression
'''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from utils.misc import plot_predictions, inverse_transform, create_interactive_plot, create_df, generate_data

def linear(save_fig=False, compare=False):
    '''
    main function for testing Linear Regression models

    Keyword Args
    ===========
    `save_fig`:
        if True, save a matplotlib figure AND plotly interactive figure
        of predictions

    `compare`:
        if true, compare values with trailer data to the baseline
    '''
    x_train, x_test, y_train, y_test, _, _, dataset, test_indices = generate_data(
        "dbs/data_2010s.csv", scale_input=False)
    # get parameters for comparing to the "baseline"
    if compare:
        # line is too long
        baseline_data = generate_data(
            "dbs/data_2010s.csv",
            drop_features=[
                'title', 'tmdb_id', 'year', 'view_count',
                'like_count', 'dislike_count', 'comment_count'],
            scale_input=False
            )
        x_train_b, x_test_b, y_train_b, y_test_b, _, _, _, _ = baseline_data

    # create instance of Linear Regression class
    regressor = LinearRegression()
    # fit the regressor to the training data
    regressor.fit(x_train, y_train)
    # use the regressor to predict the testing data
    preds = regressor.predict(x_test)
    # get metric for results comparison
    r_squared = r2_score(y_test, preds)
    print("Full: ", r_squared)

    # train and test a linear regressor on the "baseline"
    if compare:
        regressor = LinearRegression()
        regressor.fit(x_train_b, y_train_b)
        preds = regressor.predict(x_test_b)
        r_squared = r2_score(y_test_b, preds)
        print("Baseline: ", r_squared)
    
    if save_fig:
        plot_predictions(preds, y_test, r_squared, model="LinearRegression", show_fig=True)
        test_data = create_df(preds, dataset, test_indices)
        create_interactive_plot(test_data, model=f"Linear Regression")

if __name__ == "__main__":
    linear()