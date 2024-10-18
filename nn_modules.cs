// Базовый класс для всех модулей
using System.Diagnostics;

abstract class Module
{
    public int[] shape;
    public string name;
    public bool train_mode { get; set; }
    public double LR { get; set; }

    public double[] outp;
    public double[] grad;

    public abstract string info_str();
    //public abstract void set_weights();
    public abstract double[] forward(double[] X);
    public abstract double[] backward(double[] prev_out, double[] next_grad);
}

// Класс последовательности слоёв
class Sequential : Module
{
    Module[] Sequence;
    public int Nlayers;
    private bool _train_mode;

    public Dictionary<int, double[]> Grads = new Dictionary<int, double[]>();
    public Dictionary<int, double[]> Outputs = new Dictionary<int, double[]>();
    public Dictionary<int, int[]> Shapes = new Dictionary<int, int[]>();

    public Module this[int n] { get { return Sequence[n]; } }

    public Sequential(params Module[] M_sequence)
    {
        name = "Sequential";
        Sequence = M_sequence;
        Nlayers = Sequence.Length;
        train_mode = true;
        int max_shape = 0;

        for (int n = 0; n < Nlayers; n++)
        {
            Shapes.Add(n, M_sequence[n].shape);
            int temp = M_sequence[n].shape.Max();
            if (temp > max_shape) { max_shape = temp; }
        }

        shape = new int[2];
        shape[0] = Shapes[0][0];
        shape[1] = Shapes[Nlayers - 1][1];
        grad = new double[shape[0]];
        outp = new double[shape[1]];

        Random rand = new Random();
        double[] igniter = new double[max_shape];
        for (int k = 0; k < max_shape; k++) { igniter[k] = rand.NextDouble(); }

        for (int n = 0; n < Nlayers; n++)
        {
            Outputs.Add(n, igniter[..Shapes[n][1]]);
            Grads.Add(n, igniter[..Shapes[n][0]]);
        }

    }

    public Sequential(StreamReader str)
    {
        name = "Sequential";
        int m_counter = 0;
        train_mode = false;
        var s = str.ReadLine().Split(" ");

        Debug.Assert(s[0] == "Sequential", "File is not started with Sequential class info");
        Debug.Assert(s.Length == 4, "Not enough info");
        shape = new int[2];
        shape[0] = Convert.ToInt32(s[1]);
        shape[1] = Convert.ToInt32(s[2]);
        grad = new double[shape[0]];
        outp = new double[shape[1]];

        Nlayers = Convert.ToInt32(s[3]);
        Sequence = new Module[Nlayers];

        while (str.EndOfStream == false)
        {
            s = str.ReadLine().Split(" ");
            if (s[0] == "Linear")
            {
                var in_f = Convert.ToInt32(s[1]);
                var out_f = Convert.ToInt32(s[2]);
                var bias_on = Convert.ToBoolean(s[3]);
                var tmp = new Linear(in_f, out_f, bias_on);

                //double[,] tmp_w = new double[in_f, out_f];
                for (int i = 0; i < in_f; i++)
                {
                    s = str.ReadLine().Split("\t");
                    for (int j = 0; j < out_f; j++)
                        tmp.weigths[i, j] = Convert.ToDouble(s[j + 1]);
                }

                if (bias_on)
                {
                    s = str.ReadLine().Split("\t");
                    for (int j = 0; j < out_f; j++)
                        tmp.bias_w[j] = Convert.ToDouble(s[j + 1]);
                }

                Sequence[m_counter] = tmp;
                m_counter += 1;
            }

            if (s[0] == "ReLU")
            {
                Sequence[m_counter] = new ReLU(Convert.ToInt32(s[1]));
                m_counter += 1;
            }
        }

        int max_shape = 0;
        for (int n = 0; n < Nlayers; n++)
        {
            Shapes.Add(n, Sequence[n].shape);
            int temp = Sequence[n].shape.Max();
            if (temp > max_shape) { max_shape = temp; }
        }

        Random rand = new Random();
        double[] igniter = new double[max_shape];
        for (int k = 0; k < max_shape; k++) { igniter[k] = rand.NextDouble(); }

        for (int n = 0; n < Nlayers; n++)
        {
            Outputs.Add(n, igniter[..Shapes[n][1]]);
            Grads.Add(n, igniter[..Shapes[n][0]]);
        }
    }
    public override double[] forward(double[] X)
    {
        Sequence[0].forward(X);

        for (int n = 1; n < Nlayers; n++)
            Sequence[n].forward(Sequence[n - 1].outp);

        outp = Sequence[Nlayers - 1].outp;
        return outp;
    }

    public override double[] backward(double[] X0, double[] dY)
    {
        //Debug.Assert(train_mode, "Train mode is not enabled");
        Sequence[Nlayers - 1].backward(Sequence[Nlayers - 2].outp, dY);

        for (int n = Nlayers - 2; n > 0; n--)
            Sequence[n].backward(Sequence[n - 1].outp, Sequence[n + 1].grad);

        grad = Sequence[0].backward(X0, Sequence[1].grad);

        return grad;
    }

    public new double LR
    {
        set
        {
            for (int n = 0; n < Nlayers; n++) { Sequence[n].LR = value; }
        }
    }

    public new bool train_mode
    {
        set
        {
            _train_mode = value;
            for (int n = 0; n < Nlayers; n++) { Sequence[n].train_mode = _train_mode; }
        }

        get { return _train_mode; }
    }

    public override string info_str()
    {
        return $"{name} {shape[0]} {shape[1]} {Nlayers}"; ;
    }
}

// Класс полносвязного одномерного линейного слоя
class Linear : Module
{
    public double[,] weigths;
    private bool bias;
    public double[]? bias_w;

    Random rnd = new();
    public Linear(int input_size, int out_size, bool bias_on)
    {
        name = "Linear";
        shape = new int[2];
        shape[0] = input_size;
        shape[1] = out_size;
        grad = new double[input_size];
        outp = new double[out_size];
        bias = bias_on;

        if (bias)
        {
            bias_w = new double[shape[1]];
            for (int i1 = 0; i1 < shape[1]; i1++) { bias_w[i1] = 0; }
        }

        weigths = new double[shape[0], shape[1]];

        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            for (int i0 = 0; i0 < shape[0]; i0++)
                weigths[i0, i1] = G(Math.Sqrt(2.0 / shape[0]), 0);
        }
    }

    public override double[] forward(double[] X)
    {
        Debug.Assert(X.Length == shape[0], "Input shape do not match layer shape");

        for (int i1 = 0; i1 < shape[1]; i1++)
        {
            outp[i1] = 0;
            if (bias) { outp[i1] += bias_w[i1]; };

            for (int i0 = 0; i0 < shape[0]; i0++)
                outp[i1] += weigths[i0, i1] * X[i0];
        }
        return outp;
    }

    public override double[] backward(double[] X_prev, double[] dY)
    {

        Debug.Assert(X_prev.Length == shape[0], "Prev. layer grad and this layer shape[0] do not match");
        Debug.Assert(dY.Length == shape[1], "Next layer grad and this layer shape[1] do not match");

        for (int j = 0; j < shape[1]; j++)
            for (int i = 0; i < shape[0]; i++)
                grad[i] = weigths[i, j] * dY[j];

        for (int j = 0; j < shape[1]; j++)
            for (int i = 0; i < shape[0]; i++)
                weigths[i, j] -= LR * X_prev[i] * dY[j];

        if (bias)
        {
            for (int j = 0; j < shape[1]; j++)
                bias_w[j] -= LR * dY[j];
        }

        return grad;
    }

    public override string info_str()
    {
        string out_str = $"{name} {shape[0]} {shape[1]} {bias}";
        for (int i = 0; i < shape[0]; i++)
        {
            out_str += "\n";
            for (int j = 0; j < shape[1]; j++)
                out_str += $"\t{weigths[i, j]:f8}";
        }

        if (bias)
        {
            out_str += "\n";
            for (int j = 0; j < shape[1]; j++)
                out_str += $"\t{bias_w[j]:f8}";
        }
        return out_str;
    }

    double G(double stddev, double mean)
    {
        double x1 = 1 - rnd.NextDouble();
        double x2 = 1 - rnd.NextDouble();

        double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);

        return y1 * stddev + mean;
    }
}

// Класс одномерного дропаут-слоя
class Dropout1D : Module
{
    int mem = 0;
    int freq = 0;
    bool Rand = false;
    int num_off = 0;
    List<int> mask;

    public Dropout1D(int dim, int freq, int num_off)
    {
        name = "Dropout1D";
        shape = new int[2];
        shape[0] = dim;
        shape[1] = dim;
        Rand = true;
        this.freq = freq;
        this.num_off = num_off;

        grad = new double[dim];
        outp = new double[dim];

        Random rnd = new Random();

        mask = Enumerable.Range(1, dim).OrderBy(x => rnd.Next()).Take(num_off).ToList();
    }
    public Dropout1D(int dim, List<int> mask)
    {
        shape[0] = dim;
        shape[1] = dim;

        num_off = mask.Count;
        this.mask = mask;
    }

    public override double[] forward(double[] X)
    {
        outp = X;
        if (train_mode)
        {
            for (int i = 0; i < shape[0]; i++)
                if (mask.Contains(i)) { outp[i] = 0; }

            if (Rand)
            {
                mem += 1;
                if (mem == freq)
                {
                    mem = 0;
                    Random rnd = new Random();
                    mask = Enumerable.Range(1, shape[0]).OrderBy(x => rnd.Next()).Take(num_off).ToList();
                }
            }
        }
        return outp;
    }

    public override double[] backward(double[] X, double[] dY)
    {
        grad = dY;
        for (int i = 0; i < shape[0]; i++)
            if (mask.Contains(i)) { grad[i] = 0; }

        return grad;
    }

    public override string info_str()
    {
        return $"{name} {shape[0]} {shape[1]} {freq} {num_off}";
    }
}

// Класс слоя аугментации
class GaussAugment1D : Module
{
    int[] mask;
    Random rnd;
    double[] std;
    double[] mean;
    int L;

    public GaussAugment1D(int dim, int[] tgt, double[] std, double[] mean)
    {
        name = "GaussAugment1D";
        rnd = new Random();

        shape = new int[2];
        shape[0] = dim;
        shape[1] = dim;

        mask = tgt;

        this.std = std;
        this.mean = mean;
        L = mask.Length;

        grad = new double[dim];
        outp = new double[dim];
    }

    public override double[] forward(double[] X)
    {
        outp = X;
        if (train_mode)
        {
            for (int i = 0; i < L; i++)
            {
                var tmp = G(std[i], mean[i]);
                outp[mask[i]] += tmp;
            }
        }

        return outp;
    }

    public override double[] backward(double[] X, double[] dY)
    {
        grad = dY;
        return grad;
    }

    public override string info_str()
    {
        return $"{name} {shape[0]} {shape[1]}";
    }

    double G(double stddev, double mean)
    {
        double x1 = 1 - rnd.NextDouble();
        double x2 = 1 - rnd.NextDouble();

        double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);

        return y1 * stddev + mean;
    }
}


// Функции активации
class Sigmoid : Module
{
    public Sigmoid(int dim)
    {
        name = "Sigmoid";
        shape = new int[2];
        shape[0] = dim;
        shape[1] = dim;
        outp = new double[dim];
        grad = new double[dim];
    }
    public override double[] forward(double[] X)
    {
        Debug.Assert(X.Length == shape[0], "Input shape do not match layer shape");

        for (int i0 = 0; i0 < shape[0]; i0++)
            outp[i0] = 1.0 / (1 + Math.Exp(X[i0]));

        return outp;
    }

    public override double[] backward(double[] prev, double[] dY)
    {
        Debug.Assert(dY.Length == outp.Length, "Do not match lengths: outp|" + outp.Length.ToString() + " dY|" + dY.Length.ToString());

        for (int i0 = 0; i0 < shape[0]; i0++)
            grad[i0] = outp[i0] * (outp[i0] - 1) * dY[i0];

        return grad;
    }

    public override string info_str()
    {
        return $"{name} {shape[0]} {shape[1]}";
    }
}

class Tanh : Module
{
    public Tanh(int dim)
    {
        name = "Tanh";
        shape = new int[2];
        shape[0] = dim;
        shape[1] = dim;
        outp = new double[dim];
        grad = new double[dim];
    }

    public override double[] forward(double[] X)
    {
        Debug.Assert(X.Length == shape[0], "Input shape do not match layer shape");

        for (int i0 = 0; i0 < shape[0]; i0++)
            outp[i0] = Math.Tanh(X[i0]);

        return outp;
    }

    public override double[] backward(double[] X, double[] dY)
    {

        for (int i = 0; i < shape[0]; i++)
            grad[i] = (1 - outp[i] * outp[i]) * dY[i];

        return grad;
    }

    public override string info_str()
    {
        return $"{name} {shape[0]} {shape[1]}";
    }
}

class ReLU : Module
{
    public ReLU(int dim)
    {
        name = "ReLU";
        shape = new int[2];
        shape[0] = dim;
        shape[1] = dim;
        outp = new double[dim];
        grad = new double[dim];
    }

    public override double[] forward(double[] X)
    {
        Debug.Assert(X.Length == shape[0], "Input shape do not match layer shape");
        outp = new double[shape[0]];

        for (int i0 = 0; i0 < shape[0]; i0++)
            outp[i0] = Math.Max(0, X[i0]);

        return outp;
    }

    public override double[] backward(double[] X, double[] dY)
    {
        for (int i = 0; i < shape[0]; i++)
        {
            if (X[i] > 0) { grad[i] = dY[i]; }
            else { grad[i] = 0; }
        }

        return grad;
    }

    public override string info_str()
    {
        return $"{name} {shape[0]} {shape[1]}";
    }
}

class Hardsigmoid : Module
{
    public Hardsigmoid(int dim)
    {
        name = "Hardrelu";
        shape = new int[2];
        shape[0] = dim;
        shape[1] = dim;
        outp = new double[dim];
        grad = new double[dim];
    }

    public override double[] forward(double[] X)
    {
        Debug.Assert(X.Length == shape[0], "Input shape do not match layer shape");
        outp = new double[shape[0]];

        for (int i0 = 0; i0 < shape[0]; i0++)

            outp[i0] = Math.Min(Math.Max(0, X[i0] / 6 + 0.5), 1);

        return outp;
    }

    public override double[] backward(double[] X, double[] dY)
    {
        for (int i = 0; i < shape[0]; i++)
        {
            grad[i] = 0;
            if ((X[i] < 3) & (X[i] > -3)) { grad[i] = dY[i] * 1 / 6; }
        }

        return grad;
    }

    public override string info_str()
    {
        return $"{name} {shape[0]} {shape[1]}";
    }
}

class MyELU : Module
{
    double gma = 1;
    public MyELU(int dim)
    {
        name = "MyELU";
        shape = new int[2];
        shape[0] = dim;
        shape[1] = dim;
        outp = new double[dim];
        grad = new double[dim];
    }

    public override double[] forward(double[] X)
    {
        Debug.Assert(X.Length == shape[0], "Input shape do not match layer shape");
        outp = new double[shape[0]];

        for (int i0 = 0; i0 < shape[0]; i0++)
            if (X[i0] > 0) { outp[i0] = 1 - Math.Exp(-gma * X[i0]); }
            else { outp[i0] = 0; }

        return outp;
    }

    public override double[] backward(double[] X, double[] dY)
    {
        for (int i = 0; i < shape[0]; i++)
        {
            if (X[i] > 0) { grad[i] = dY[i] * gma * (1 - outp[i]); }
            else { grad[i] = 0; }
        }

        return grad;
    }

    public override string info_str()
    {
        return $"{name} {shape[0]} {shape[1]}";
    }
}

class MySHR : Module
{
    double x_lim = 1;
    double neg_leak = 0.01;
    double pos_leak = 0.01;
    public MySHR(int dim)
    {
        name = "MySHR";
        shape = new int[2];
        shape[0] = dim;
        shape[1] = dim;
        outp = new double[dim];
        grad = new double[dim];
    }

    public override double[] forward(double[] X)
    {
        Debug.Assert(X.Length == shape[0], "Input shape do not match layer shape");
        outp = new double[shape[0]];

        for (int i0 = 0; i0 < shape[0]; i0++)
            if (X[i0] < 0) { outp[i0] = Math.Max(-neg_leak * X[i0], 0); }
            else
            {
                if (X[i0] > x_lim) { outp[i0] = Math.Max(1 - pos_leak * X[i0], 0); }
                else { outp[i0] = X[i0] / x_lim; }
            }

        return outp;
    }

    public override double[] backward(double[] X, double[] dY)
    {
        for (int i = 0; i < shape[0]; i++)
        {
            outp[i] = 0;
            if ((X[i] < 0) & (X[i] > -1 / neg_leak)) { outp[i] = neg_leak; }
            else
            {
                if (X[i] < x_lim) { outp[i] = 1 / x_lim; }
                else
                {
                    if (X[i] < (x_lim + 1 / pos_leak)) { outp[i] = pos_leak; }
                }
            }
        }

        return grad;
    }

    public override string info_str()
    {
        return $"{name} {shape[0]} {shape[1]}";
    }
}
