#include <bits/stdc++.h>
#include <fstream>
using namespace std;

int main()
{

    for (int x = 1; x <= 10000; x++)
    {
        double val = sin((tan((log2(x) * x)))) + atan(log2(x * (1 / cos(30)))) + (x * exp(tan(cos(x) * sin((x * 1.43)))));
        cout << x << " , " << val << endl;
    }

    return 0;
}