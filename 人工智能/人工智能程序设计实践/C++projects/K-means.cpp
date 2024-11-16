#include <iostream> 
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <string>
#include <math.h>
#include <stdlib.h>
using namespace std;

struct Flower{
	int id;
	double character1;
    double character2;
    double character3;
    double character4;
	string labels;
	int label = 0;
};

struct Distance{
	struct Flower information;
	int labels;
};


// 初始化中心点
void GenerateCenters(int k, vector<struct Flower> FlowerVector, vector<struct Flower> &random_centers)
{
	vector<struct Flower> tmp(FlowerVector);

	int random_index;
	int n = FlowerVector.size();
	
	random_index = rand() % (50-0) + 0;
	//cout << random_index << endl;
	random_centers.push_back(FlowerVector.at(random_index));

	random_index = rand() % (100-50) + 50;
	random_centers.push_back(FlowerVector.at(random_index));

	random_index = rand() % (150-100) + 100;
	random_centers.push_back(FlowerVector.at(random_index));
}

double GetDistance(Flower p0, Flower p1)
{
	return sqrt((p0.character1 - p1.character1)*(p0.character1 - p1.character1) + (p0.character2 - p1.character2)*(p0.character2 - p1.character2) + (p0.character3 - p1.character3)*(p0.character3 - p1.character3) + (p0.character4 - p1.character4)*(p0.character4 - p1.character4));
}

void DoKmeansCluster(vector<struct Flower> FlowerVector, vector<struct Flower> &random_centers, vector<struct Distance> &result)
{
	vector<struct Flower> tmp(random_centers);

	int point_num = FlowerVector.size();
	int k = random_centers.size();

	//根据各点到中心点的距离聚类
	

	
	for (int p = 0; p < point_num; p++)
	{
		float distance = 9999;
		Distance res;
		for (int q = 0; q < k; q++)
		{
			float tmp_distance = GetDistance(FlowerVector[p], random_centers[q]);
			if (tmp_distance < distance)
			{
				distance = tmp_distance;
				res.labels = q;
				res.information = FlowerVector[p];
			}
		}
		result.push_back(res);
	}

	//根据聚类结果更新中心点
	for (int i = 0; i < k; i++)
	{
		int count = 0;
		double sum_1 = 0;
		double sum_2 = 0;
		double sum_3 = 0;
		double sum_4 = 0;
		for (int j = 0; j < point_num; j++)
		{
			if (result[j].labels == i)
			{
				count++;
				sum_1 += result[j].information.character1;
				sum_2 += result[j].information.character2;
				sum_3 += result[j].information.character3;
				sum_4 += result[j].information.character4;
			}
		}
		random_centers[i].character1 = sum_1 / count;
		random_centers[i].character2 = sum_2 / count;
		random_centers[i].character3 = sum_3 / count;
		random_centers[i].character4 = sum_4 / count;
		//cout << "(" << random_centers[i].x << "," << random_centers[i].y << ")" << endl;
	}


}



int main()
{
    ifstream infile("Iris.csv", ios::in);
    string line;
	vector<struct Flower> FlowerVector;
    getline(infile, line);
    	while (getline(infile, line)) 
		{
		stringstream ss(line);
		string str;
		Flower flower;

		getline(ss, str, ',');
		flower.id = stoi(str);
		getline(ss, str, ',');
		flower.character1 = stod(str);
		getline(ss, str, ',');
		flower.character2 = stod(str);
		getline(ss, str, ',');
		flower.character3 = stod(str);
		getline(ss, str, ',');
		flower.character4 = stod(str);
        getline(ss, str, ',');
		flower.labels = str;
		FlowerVector.push_back(flower);
		}
    int x = FlowerVector.size();
		for (int i = 0; i < x; i++)
		{
		if (FlowerVector[i].labels == "Iris-setosa")
			FlowerVector[i].label = 1;
		if (FlowerVector[i].labels == "Iris-versicolor")
			FlowerVector[i].label = 2;
		if (FlowerVector[i].labels == "Iris-virginica")
			FlowerVector[i].label = 3;
		}
	int k = 3;
	int epochs = 500;
	
	vector<Flower> random_centers;
	GenerateCenters(k, FlowerVector, random_centers);
	vector<struct Distance> result;
	for(int i=0;i<epochs;i++)
	{
		DoKmeansCluster(FlowerVector, random_centers, result);
	}
	//打印中心坐标--4维对应四个特征向量
	for(int i=0;i<k;i++)
		cout << "(" << random_centers[i].character1 << "," << random_centers[i].character2 << "," << random_centers[i].character3 << "," << random_centers[i].character4 << ")" << endl;
	//打印各数据对应标签，1为Iris-setosa，2为Iris-versicolor，3为Iris-virginica
	for(int i=0;i<150;i++)
		cout<<i+1<<','<<result[i].labels+1<<endl;
}