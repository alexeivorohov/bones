// Функции ошибок
public abstract class LossF
{
    public abstract double eval(double[] Y, double[] Y_true);

    public abstract double[] grad(double[] X, double[] Y_true);
}

public class R2 : LossF
{
    public override double eval(double[] Y, double[] Y_true)
    {
        int N = Y.Length;
        double res = 0;

        for (int i = 0; i < N; i++)
            res += Math.Pow((Y[i] - Y_true[i]), 2);

        return (res / N);
    }

    public override double[] grad(double[] Y, double[] Y_true)
    {
        int N = Y.Length;
        double[] res = new double[N];

        for (int i = 0; i < N; i++)
            res[i] = 2 * (Y[i] - Y_true[i]);

        //Console.WriteLine($"Inp: {Y[0]:f8} LossGrad {res[0]:f8}");
        return res;
    }
}

public class CrossEntropy : LossF
{
    public override double eval(double[] Y, double[] Y_true)
    {
        int N = Y.Length;
        double res = 0;

        for (int i = 0; i < N; i++)
            res += -(Y_true[i] * Math.Log(Y[i]) + (1 - Y_true[i]) * Math.Log(1 - Y[i])); //(1-Y[i])*Y_true[i] + (1 - Y_true[i])*Y[i];

        return (res / N);
    }

    public override double[] grad(double[] Y, double[] Y_true)
    {
        int N = Y.Length;
        double[] res = new double[N];

        for (int i = 0; i < N; i++)
            res[i] = -Y_true[i] + Y[i];

        //Console.WriteLine($"Inp: {Y[0]:f8} LossGrad {res[0]:f8}");
        return res;
    }
}

// Класс функций оценки
public abstract class Score
{
    public abstract double[] Eval(DataFrame Y, DataFrame Y_true);
    public abstract double Eval1(double[] Y, double[] Y_true);
}

class MRE : Score
{
    public override double[] Eval(DataFrame Y, DataFrame Y_true)
    {
        int n = Y.shape[0];
        double[] res = new double[n + 1];
        for (int i = 0; i < n; i++)
        {
            res[i] = Eval1(Y[i], Y_true[i]);
            res[^1] += res[i];
        }

        res[^1] /= n;

        return res;
    }

    public override double Eval1(double[] y, double[] y_true)
    {
        int n = y_true.Length;
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += Math.Abs((y[i] - y_true[i]) / y_true[i]);
        }
        sum /= n;
        return sum;
    }

}

class MAE : Score
{
    public override double[] Eval(DataFrame Y, DataFrame Y_true)
    {
        int n = Y.shape[0];
        double[] res = new double[n + 1];
        for (int i = 0; i < n; i++)
        {
            res[i] = Eval1(Y[i], Y_true[i]);
            res[^1] += res[i];
        }

        res[^1] /= n;

        return res;
    }

    public override double Eval1(double[] y, double[] y_true)
    {
        int n = y_true.Length;
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += Math.Abs(y[i] - y_true[i]);
        }
        sum /= n;
        return sum;
    }
}

class RMSE : Score
{
    public override double[] Eval(DataFrame Y, DataFrame Y_true)
    {
        int n = Y.shape[0];
        double[] res = new double[n + 1];
        for (int i = 0; i < n; i++)
        {
            res[i] = Eval1(Y[i], Y_true[i]);
            res[^1] += res[i];
        }

        res[^1] /= n;

        return res;
    }

    public override double Eval1(double[] y, double[] y_true)
    {
        int n = y_true.Length;
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += Math.Pow((y[i] - y_true[i]), 2);
        }
        sum = Math.Sqrt(sum) / n;
        return sum;
    }
}

class Accuracy : Score
{
    public double b2 = 1;
    double tol = 0.5;

    public override double[] Eval(DataFrame Y, DataFrame Y_true)
    {
        int n = Y.shape[0];
        double[] res = new double[n + 1];
        for (int i = 0; i < n; i++)
        {
            res[i] = Eval1(Y[i], Y_true[i]);
            res[^1] += res[i];
        }

        res[^1] /= n;

        return res;
    }

    public override double Eval1(double[] y, double[] y_true)
    {
        int L = y.Length;
        double fn = 0;
        double fp = 0;
        double tptn = 0;
        double lbl;


        for (int j = 0; j < L; j++)
        {
            lbl = Math.Round(y[j]) - y_true[j];

            if (lbl == 0)
                tptn += 1;

            if (lbl < 0)
                fn += 1;

            if (lbl > 0)
                fp += 1;

        }


        return tptn / (tptn + fn + fp);

    }
}

class Fb_score : Score
{
    public double b2 = 1;
    double tol = 0.5;

    public override double[] Eval(DataFrame Y, DataFrame Y_true)
    {
        int n = Y.shape[0];
        int L = Y.shape[1];
        double fn = 0;
        double fp = 0;
        double tp = 0;
        double lbl;
        double[] res = new double[n + 1];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < L; j++)
            {
                lbl = Math.Round(Y[i][j]);

                if (lbl == 1)
                    if (Y_true[i][j] == 1) { tp += 1; }
                    else { fp += 1; }

                if ((lbl == 0) & (Y_true[i][j] == 1))
                    fn += 1;

            }

        }

        res[^1] = (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp);

        return res;
    }

    public override double Eval1(double[] y, double[] y_true)
    {
        int L = y.Length;
        double fn = 0;
        double fp = 0;
        double tp = 0;
        double lbl;


        for (int j = 0; j < L; j++)
        {
            lbl = Math.Round(y[j]);

            if (lbl == 1)
                if (y_true[j] == 1) { tp += 1; }
                else { fp += 1; }

            if ((lbl == 0) & (y_true[j] == 1))
                fn += 1;

        }

        return (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp);

    }

}
