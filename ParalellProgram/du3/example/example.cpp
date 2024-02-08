
#include <utility>
#include <vector>
#include <iostream>
#include <cstring>
#include <omp.h>
#include <cassert>

class EditDistance
{
private:
    using C = char;
    using DIST = std::size_t;
    bool DEBUG = false;
    std::vector<DIST> array_1;
    std::vector<DIST> array_2;
    std::vector<DIST> array_3;

    std::vector<DIST> *first_diagonal;
    std::vector<DIST>(*second_diagonal);
    std::vector<DIST>(*third_diagonal);

    DIST width;
    DIST height;

    bool isStr1Shorter = false;

    void moveToNextDiagonal()
    {
        std::vector<DIST> *new_first = third_diagonal;
        third_diagonal = second_diagonal;
        second_diagonal = first_diagonal;
        first_diagonal = new_first;
    }

    DIST min(const DIST first, const DIST second, const DIST third)
    {
        if (first < second && first < third)
            return first;
        else if (second < third)
            return second;
        else
            return third;
    }

    void handleOneDiagonal(const DIST abstractDiagonalSize, const DIST diagonalSize, const std::vector<C> &str1, const std::vector<C> &str2)
    {
        DIST posInStr1 = 0;
        DIST posInStr2 = abstractDiagonalSize - 2;

#pragma omp parallel for
        for (DIST i = 1; i < diagonalSize; i++)
        {
            bool charEquals = str1[posInStr1 + i - 1] == str2[posInStr2 - i + 1];
            (*first_diagonal)[i] = min(
                (*second_diagonal)[i - 1] + 1,
                (*second_diagonal)[i] + 1,
                (*third_diagonal)[i - 1] + !charEquals);
        }
    }

    void handleOneReduceDiagonalSpecialLine(
        const DIST diagonalSize, const std::vector<C> &str1, const std::vector<C> &str2)
    {
        DIST posInStr1 = width - diagonalSize;

        // -1 to skip first position, + -1 for the size to index conversion
        DIST posInStr2 = height - 1;

        // avoid first item -> i != 0
#pragma omp parallel for
        for (DIST i = 0; i < diagonalSize; i++)
        {
            bool charEquals = str1[posInStr1 + i] == str2[posInStr2 - i];
            (*first_diagonal)[i] = min(
                (*second_diagonal)[i] + 1,
                (*second_diagonal)[i + 1] + 1,
                (*third_diagonal)[i] + !charEquals);
        }
    }

    void handleOneReduceDiagonal(
        const DIST diagonalSize, const std::vector<C> &str1, const std::vector<C> &str2)
    {
        DIST posInStr1 = width - diagonalSize;

        // -1 to skip first position, + -1 for the size to index conversion
        DIST posInStr2 = height - 1;

        // avoid first item -> i != 0
#pragma omp parallel for
        for (DIST i = 0; i < diagonalSize; i++)
        {
            bool charEquals = str1[posInStr1 + i] == str2[posInStr2 - i];
            (*first_diagonal)[i] = min(
                (*second_diagonal)[i] + 1,
                (*second_diagonal)[i + 1] + 1,
                (*third_diagonal)[i + 1] + !charEquals);
        }
    }

    DIST catchSpecialCases(const std::vector<C> &str1, const std::vector<C> &str2){
        if(width == 0)
            return height;

        return SIZE_MAX;
    }

    DIST computeSortedStrings(const std::vector<C> &str1, const std::vector<C> &str2)
    {
        DIST specialCase = catchSpecialCases(str1, str2);
        if(specialCase != SIZE_MAX)
            return specialCase;

        DIST diagonal = 0;
        DIST diagonalSize = 2;

        // diagonals are growing
        for (; diagonal < width - 1; ++diagonal)
        {
            (*second_diagonal)[0] = diagonalSize - 1;
            (*second_diagonal)[diagonal + 1] = diagonalSize - 1;
            handleOneDiagonal(diagonalSize, diagonalSize, str1, str2);
            ++diagonalSize;
            moveToNextDiagonal();
        }

        (*second_diagonal)[0] = diagonalSize - 1;
        (*second_diagonal)[diagonal + 1] = diagonalSize - 1;

        DIST lastGrowingDiagonal = diagonal;
        
        // diagonals has constant size of the width
        DIST abstractDiagonalSize = diagonalSize;
        for (; diagonal < lastGrowingDiagonal + height - width; ++diagonal)
        {
            (*second_diagonal)[0] = abstractDiagonalSize - 1;
            handleOneDiagonal(abstractDiagonalSize, diagonalSize, str1, str2);
            moveToNextDiagonal();
            ++abstractDiagonalSize;
        }

        (*second_diagonal)[0] = abstractDiagonalSize - 1;

        DIST lastConstantDiagonal = diagonal;

        // diagonals are shrinking
        --diagonalSize;
        handleOneReduceDiagonalSpecialLine(diagonalSize, str1, str2);
        moveToNextDiagonal();

        for (; diagonal < lastConstantDiagonal + width; diagonal++)
        {
            --diagonalSize;
            handleOneReduceDiagonal(diagonalSize, str1, str2);
            moveToNextDiagonal();
        }

        return (*third_diagonal)[0];
    }

public:
    void init(DIST len1, DIST len2)
    {
        if (len1 < len2)
        {
            isStr1Shorter = true;
        }
        width = (std::size_t)std::min<DIST>(len1, len2);
        height = (std::size_t)std::max<DIST>(len1, len2);
        array_1.resize(width + 2);
        array_2.resize(width + 2);
        array_3.resize(width + 2);

        first_diagonal = &array_1;
        second_diagonal = &array_2;
        third_diagonal = &array_3;
        (*third_diagonal)[0] = 0;
    }

    DIST compute(const std::vector<C> &str1, const std::vector<C> &str2)
    {
        if (isStr1Shorter)
            return computeSortedStrings(str1, str2);
        else
            return computeSortedStrings(str2, str1);
    }
};


size_t count(const char *cstr1, const char *cstr2)
{
/*  */    std::vector<char> str1(cstr1, cstr1 + std::strlen(cstr1));
    std::vector<char> str2(cstr2, cstr2 + std::strlen(cstr2));

    EditDistance edit;
    edit.init(str1.size(), str2.size());

    size_t out = edit.compute(str1, str2);
    std::cout << out << std::endl;
    return out;
}

int main(int argc, char const *argv[])
{
    
    const char *cstr1 = "";
    const char *cstr2 = "aa";

    assert(count(cstr1, cstr2) == 2);
    
    cstr1 = "a";
    cstr2 = "aaa";
    assert(count(cstr1, cstr2) == 2);

    cstr1 = "aa";
    cstr2 = "aaaa";
    assert(count(cstr1, cstr2) == 2);
    
    cstr1 = "aaa";
    cstr2 = "aaaaa";
    assert(count(cstr1, cstr2) == 2);
    
    cstr1 = "ab";
    cstr2 = "aa";
    assert(count(cstr1, cstr2) == 1);
    
    cstr1 = "aba";
    cstr2 = "aaa";
    assert(count(cstr1, cstr2) == 1);
    
    cstr1 = "ababababa";
    cstr2 = "aaaaaaaaa";
    assert(count(cstr1, cstr2) == 4);
    
    cstr1 = "ababa";
    cstr2 = "aaaaaaaaa";
    assert(count(cstr1, cstr2) == 6);
    
    cstr1 = "abab";
    cstr2 = "aaaaaaaaa";
    assert(count(cstr1, cstr2) == 7);
    
    cstr1 = "aaaa";
    cstr2 = "ababababa";
    assert(count(cstr1, cstr2) == 5);
    
    cstr1 = "a";
    cstr2 = "ba";
    assert(count(cstr1, cstr2) == 1);
    
    cstr1 = "abc";
    cstr2 = "bac";
    assert(count(cstr1, cstr2) == 2);

    cstr1 = "aa";
    cstr2 = "bbaa";
    assert(count(cstr1, cstr2) == 2);
    
    return 0;
}