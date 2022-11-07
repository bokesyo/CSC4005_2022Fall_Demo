#include <iostream>
#include <vector>

int main() {
    std::vector<char> data(102400, 'A');

#pragma omp parallel for shared(data) default(none)
    for (int i = 0; i < data.size(); i++) {
        data[i] ^= 'A' ^ 'a';
    }

    for (auto &i : data) {
        std::cout << i;
    }

    std::cout << std::endl;
    return 0;
}
