#include <vector>
#include <iostream>

double dot2d(const std::vector<std::vector<double>> &v1, const std::vector<std::vector<double>> &v2)
{
    if ((v1.size() != v2.size()) & (v1[0].size() != v2[0].size()))
    {
        throw "Vectors must have the same size";
    }

    double sum = 0;

    for (int i = 0; i < v1.size(); i++)
    {
        for (int j = 0; j < v1[0].size(); j++)
        {
            sum += v1[i][j] * v2[i][j];
        }
    }
    return sum;
}

std::vector<std::vector<double>> slice(const std::vector<std::vector<double>> &v, int start_row, int end_row, int start_col, int end_col)
{
    start_row = start_row < 0 ? 0 : start_row;
    start_col = start_col < 0 ? 0 : start_col;
    end_row = end_row > v.size() ? v.size() : end_row;
    end_col = end_col > v[0].size() ? v[0].size() : end_col;

    std::vector<std::vector<double>> res(end_row - start_row, std::vector<double>(end_col - start_col));

    for (int i = 0; i < res.size(); i++)
    {
        for (int j = 0; j < res[0].size(); j++)
        {
            res[i][j] = v[start_row + i][start_col + j];
        }
    }
    return res;
}

void convolve(const std::vector<std::vector<double>> &img, std::vector<std::vector<double>> &mask, std::vector<std::vector<double>> &res)
{
    int img_rows = img.size();
    int img_cols = img[0].size();
    int mask_rows = mask.size();
    int mask_cols = mask[0].size();

    for (int row = 0; row < (img_rows - mask_rows + 1); row++)
    {
        for (int col = 0; col < (img_cols - mask_cols + 1); col++)
        {
            std::vector<std::vector<double>> sub = slice(img, row, row + mask_rows, col, col + mask_cols);
            res[row][col] = dot2d(sub, mask);
        }
    }
}

void print_vec2d(std::vector<std::vector<double>> &v)
{
    std::cout << "{\n";
    for(int i = 0; i < v.size(); i++) 
    {
        std::cout << " { ";
        for (int j = 0; j < v[i].size(); j++) 
        { 
            std::cout << v[i][j] << " "; 
        }     
        std::cout << "}\n"; 
    }
    std::cout << "}\n";
}

int main(int argc, char const *argv[])
{
    // 3x3 vector with default value of 3
    std::vector<std::vector<double>> mask(3, std::vector<double>(3,3));

    // input image with default value of 1
    std::vector<std::vector<double>> img(5, std::vector<double>(5,1));

    // output vector
    std::vector<std::vector<double>> ouptut(img.size() - mask.size() + 1, std::vector<double>(img[0].size() - mask[0].size() + 1));

    convolve(img, mask, ouptut);

    print_vec2d(ouptut);

    return 0;
}
