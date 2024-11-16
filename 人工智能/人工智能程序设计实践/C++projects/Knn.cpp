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

struct Flower {
	int id;
	double character[4];
	string labels;
};

//训练集与测试集的随机分割
void GenerateTests(int m, vector<struct Flower> FlowerVector, vector<struct Flower> &random_tests,vector<struct Flower> &random_trains)
{
	vector<struct Flower> tmp(FlowerVector);

	int random_index;
	int n = FlowerVector.size();
	int mark[150]={0};
	int j = 0;
	for(int i=0;i<m;i++)
	{
		random_index = rand() % 150;
		random_tests.push_back(FlowerVector.at(random_index));
		mark[random_index] = random_index;
	}
	
	for(int i=0;i<150;i++)
	{
		if(mark[i] == 0 && j<100)
		{
			random_trains.push_back(FlowerVector.at(i));
			j++;
		}
	}
}

//计算欧式距离
double GetDistance(Flower p0, Flower p1)
{
	double sum = 0;
	for(int i=0;i<4;i++)
	{
		sum += (p0.character[i]-p1.character[i])*(p0.character[i]-p1.character[i]);
	}
	sum = sqrt(sum);
	return sum;
}

void Knn(vector<struct Flower> random_tests, vector<struct Flower> random_trains, int k )
{
	double temp;
	int a=0,b=0,c=0;
	for(int i=0;i<random_tests.size();i++)
	{
		double distance[random_trains.size()][2] = {0};
		for(int j=0;j<random_trains.size();j++)
		{
			distance[j][0] = j;
			distance[j][1] = GetDistance(random_tests[i],random_trains[j]);
			//cout<<distance[j][1]<<endl;
		}

		for(int j=1;j<random_trains.size();j++)
			for(int l=0;l<random_trains.size()-j;l++)
			{
				if(distance[l][1]>distance[l+1][1])
				{
					temp = distance[l][1];
					distance[l][1] = distance[l+1][1];
					distance[l+1][1] = temp;

					temp = distance[l][0];
					distance[l][0] =distance[l+1][0];
					distance[l+1][0] = temp;
				}
			}
	
		for(int j=0;j<k;j++)
		{

			if(random_trains[distance[j][0]].labels == "Iris-setosa")
				a++;
			if(random_trains[distance[j][0]].labels == "Iris-versicolor")
				b++;
			if(random_trains[distance[j][0]].labels == "Iris-virginica")
				c++;
			
		}
		if(a>=b && a>=c)
		{
			cout<<i+1<<','<<random_tests[i].labels<<','<<'1'<<endl;
		}
		if(b>=a && b>=c)
		{
			cout<<i+1<<','<<random_tests[i].labels<<','<<'2'<<endl;
		}
		if(c>=b && c>=a)
		{
			cout<<i+1<<','<<random_tests[i].labels<<','<<'3'<<endl;
		}
		
		a=0,b=0,c=0;
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
		flower.character[0] = stod(str);
		getline(ss, str, ',');
		flower.character[1] = stod(str);
		getline(ss, str, ',');
		flower.character[2] = stod(str);
		getline(ss, str, ',');
		flower.character[3] = stold(str);
		getline(ss, str, ',');
		flower.labels = str;
		FlowerVector.push_back(flower);
	}
	int x = FlowerVector.size();

	int m=50;
	int k=7;
	vector<struct Flower> random_tests;
	vector<struct Flower> random_trains;
	GenerateTests(m,FlowerVector, random_tests, random_trains);
	Knn(random_tests, random_trains, k );
	return 0;
}
