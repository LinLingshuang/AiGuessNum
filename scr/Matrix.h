#pragma once
#include <vector>
#include<cmath>
#include "VectorD.h"
class Matrix {
public:
	Matrix() {
		columnNum = 1;
		rowNum = 1;
		// 分配内存
		columnVector = static_cast<VectorD*>(operator new[](sizeof(VectorD) * 1));
		// 直接构造对象
		new (&columnVector[0]) VectorD(1);
	}
	Matrix(int rrowNum, int ccolumnNum, double value = 0) {
		if (rrowNum <= 0) rrowNum = 1;
		if (ccolumnNum <= 0) ccolumnNum = 1;
		rowNum = rrowNum;
		columnNum = ccolumnNum;
		// 分配内存
		columnVector = static_cast<VectorD*>(operator new[](sizeof(VectorD)* columnNum));
		// 直接构造对象
		for (int i = 0; i < columnNum; i++) {
			new (&columnVector[i]) VectorD(rowNum, value);
		}
	}
	Matrix(const Matrix& other) {
		rowNum = other.rowNum;
		columnNum = other.columnNum;
		// 分配内存
		columnVector = static_cast<VectorD*>(operator new[](sizeof(VectorD)* columnNum));
		// 直接构造对象
		for (int i = 0; i < columnNum; i++) {
			new (&columnVector[i]) VectorD(other.columnVector[i]);
		}
	}
	Matrix& operator=(const Matrix& other) {
		if (this != &other) {
			// 手动调用析构函数
			for (int i = 0; i < columnNum; i++) {
				columnVector[i].~VectorD();
			}
			// 释放内存
			operator delete[](columnVector);

			rowNum = other.rowNum;
			columnNum = other.columnNum;
			// 分配内存
			columnVector = static_cast<VectorD*>(operator new[](sizeof(VectorD)* columnNum));
			// 直接构造对象
			for (int i = 0; i < columnNum; i++) {
				new (&columnVector[i]) VectorD(other.columnVector[i]);
			}
		}
		return *this;
	}
	~Matrix() {
		// 手动调用析构函数
		for (int i = 0; i < columnNum; i++) {
			columnVector[i].~VectorD();
		}
		// 释放内存
		operator delete[](columnVector);
	}


	void setValue(int i, int j, double num) { //注意这里i、j范围要求>0
		if (i < 1 || i > rowNum) return;  // i 从1开始
		if (j < 1 || j > columnNum) return;
		else {
			columnVector[j - 1][i - 1] = num;
		}
	}
	void show(const string testname = "testname") {
		cout << endl;
		cout << testname;
		cout << endl;
		for (int i = 0; i < rowNum; i++) {
			for (int j = 0; j < columnNum; j++) {
				cout << columnVector[j][i] << ' ';
			}
			cout << endl;
		}
		cout << endl;
	}

	int getrowNum() { return rowNum; }
	int getcolumnNum() { return columnNum; }
	VectorD* getVectorD() { return columnVector; }
private:
	VectorD* columnVector;
	int rowNum;
	int columnNum;
};
