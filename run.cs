// Объявляем заголовок


string[] df_cols = {"Density", "E.[%]", "P.[%]",
            "YM", "YS", "HV", "at1", "at2", "at3", "at4", "at5", "at6",
            "at7", "at8", "Long", "Short", "Flat", "Comp."};

// Читаем данные:

Console.WriteLine("\nReading data...\n");
DataFrame df_train = new(df_cols);

df_train.ReadCSV("F_train.csv", false, " ");

df_train.Info();

int[] x_cols = { 0, 1, 2, 3, 4, 5 };
int[] at_cols = { 6, 7, 8, 9, 10, 11, 12, 13 };
int[] y_cols = { 14, 15, 16, 17};

// Предобработка данных:

var X_df = df_train.GetCols(x_cols);
MaxScaler Scl1 = new(X_df, false, 1);
var X_t = Scl1.Transform(X_df);

var at_df = df_train.GetCols(at_cols);
double[] at_norm = { 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008 };

at_df.ColumnwiseMul(at_norm);

X_t = X_t.Concat(at_df, 1);

X_t.Head(3);

var y_t = df_train.GetCols(y_cols);


// Задание модели:

double[] G0_std = { 0.03, 0.03, 0.03, 0.03, 0.03, 0.03 };
double[] G0_mean = { 0, 0, 0, 0, 0, 0 };
GaussAugment1D G0 = new(14, x_cols, G0_std, G0_mean);
Linear L1 = new(X_t.shape[1], 14, false);
ReLU A1 = new(14);
Linear L2 = new(14, 8, true);
Dropout1D D2 = new(8, 20, 2);
ReLU A2 = new(8);
Linear L3 = new(8, y_t.shape[1], true);
Sigmoid Out = new(y_t.shape[1]);

Sequential seq = new(G0, L1, A1, L2, A2, L3, Out);

LossF loss_fn = new CrossEntropy();

NN_Model Net1 = new(seq, loss_fn);


// Проверка:

// Прогрев:
Net1.fit(X_t, y_t, 200, 0.00001, null);

// Обучение:
Fb_score scorer = new();

Net1.fit(X_t, y_t, 300, 0.05, scorer);
Net1.fit(X_t, y_t, 300, 0.02, scorer);

Console.WriteLine($"\n Model best score: {Net1.best_score:f6}");


// Тестирование:
Console.WriteLine($"\n Тестирование");

DataFrame df_tst = new(df_cols);
df_tst.ReadCSV("F_test.csv", false, " ");
X_t = Scl1.Transform(df_tst);

at_df = df_tst.GetCols(at_cols);
at_df.ColumnwiseMul(at_norm);

X_t = X_t.Concat(at_df, 1);

X_t.Info();
X_t.Head(3);
y_t = df_tst.GetCols(y_cols);

Net1.eval_mode();
var y_pred = Net1.predict(X_t);


Console.WriteLine($"Test score: {scorer.Eval(y_pred, y_t)[^1]:f6}");

y_pred.Info();
y_t.Info();
var compare = y_pred.Concat(y_t, axis:1);
//compare.Info();

compare.Head(20);

Net1.DebugForward(X_t[0]);

var tst_pred = Net1.predict(X_t[0]);

var tst = loss_fn.eval(tst_pred, y_t[0]);
Console.WriteLine(tst);

var tst2 = loss_fn.grad(tst_pred, y_t[0]);
Net1.DebugBackward(X_t[0], tst2);

Net1.SaveState("Net1_tst.txt");