#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvInferPlugin.h"
#include "polyiou.h"
#include "logging.h"
#include <opencv2/opencv.hpp>
#include "utils.h"
using namespace nvinfer1;
using namespace cv;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 512;
static const int INPUT_W = 512;
static const int CLASSES = 12;
static const int Num_box = 500;
static const int OUTPUT_SIZE = Num_box *CLASSES;
static const int down_ratio = 4;

static const float CONF_THRESHOLD = 0.1;
static const float NMS_THRESHOLD = 0.5;

const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";

static Logger gLogger;
void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));//
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
	// cudaMalloc�����ڴ� cudaFree�ͷ��ڴ� cudaMemcpy�� cudaMemcpyAsync ���������豸֮�䴫������
	// cudaMemcpy cudaMemcpyAsync ��ʽ���������� ��ʽ�ط��������� 
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE  * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}



int main(int argc, char** argv)
{
	if (argc < 2){
		argv[1] = "../../model.engine";
		argv[2] = "../../samples/test1.jpg";
	}
    // create a model using the API directly and serialize it to a stream
	char* trtModelStream{ nullptr }; //char* trtModelStream==nullptr;  ���ٿ�ָ��� Ҫ��new���ʹ�ã�����89�� trtModelStream = new char[size]
    size_t size{0};//��int�̶��ĸ��ֽڲ�ͬ������ͬ,size_t��ȡֵrange��Ŀ��ƽ̨�������ܵ�����ߴ�,һЩƽ̨��size_t�ķ�ΧС��int��������Χ,�ֻ��ߴ���unsigned int. ʹ��Int���п����˷ѣ����п��ܷ�Χ������
    
    std::ifstream file(argv[1], std::ios::binary);
    if (file.good()) {
		std::cout<<"load engine success"<<std::endl;
        file.seekg(0, file.end);//ָ���ļ�������ַ
        size = file.tellg();//���ļ����ȸ��߸�size
		//std::cout << "\nfile:" << argv[1] << " size is";
		//std::cout << size << "";
		
		file.seekg(0, file.beg);//ָ���ļ��Ŀ�ʼ��ַ
        trtModelStream = new char[size];//����һ��char �������ļ��ĳ���
		assert(trtModelStream);//
        file.read(trtModelStream, size);//���ļ����ݴ���trtModelStream
        file.close();//�ر�
    }
	else {
		std::cout << "load engine failed" << std::endl;
		return 1;
	}
	
	Mat src=imread(argv[2]);
	if (src.empty()) {std::cout << "image load faild" << std::endl;return 1;}
	int img_width = src.cols;
	int img_height = src.rows;

    // Subtract mean from image
    static float data[3 * INPUT_H * INPUT_W];
	Mat pr_img0, pr_img;
	//pr_img=preprocess_img(src, INPUT_H, INPUT_W);       // Resize
	cv::resize(src, pr_img,Size(512,512), 0, 0, cv::INTER_LINEAR);
	int i = 0;// [1,3,INPUT_H,INPUT_W]
	//std::cout << "pr_img.step" << pr_img.step << std::endl;
	for (int row = 0; row < INPUT_H; ++row) {
		uchar* uc_pixel = pr_img.data + row * pr_img.step;//pr_img.step=widthx3 ����ÿһ����width��3ͨ����ֵ
		for (int col = 0; col < INPUT_W; ++col)
		{
			data[i] = (float)uc_pixel[2] / 255.0-0.5;
			data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0 - 0.5;
			data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0 - 0.5;
			uc_pixel += 3;
			++i;
		}
	}

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
	bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[OUTPUT_SIZE];
    for (int i = 0; i < 10; i++) {//����10�ε������ٶ�
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

	// �������� �����500��box  ÿ��box����12����ǰ2�������ĵ� Ȼ��8���ǿ��ĸ��ֳ��� ʣ�����ֱ������ŶȺ�cls
	std::map<float, std::vector<Detection>> m;
	auto start = std::chrono::system_clock::now();
	for (int position = 0; position < Num_box; position++) {
		float *row = prob + position * CLASSES;
		//��Щ����ԭpython���������
		float cen_pt_0 = row[0];
		float cen_pt_1 = row[1];
		float tt_2 = row[2];
		float tt_3 = row[3];
		float rr_4 = row[4];
		float rr_5 = row[5];
		float bb_6 = row[6];
		float bb_7 = row[7];
		float ll_8 = row[8];
		float ll_9 = row[9];
		float tl_0 = tt_2 + ll_8 - cen_pt_0;
		float tl_1 = tt_3 + ll_9 - cen_pt_1;
		float bl_0 = bb_6 + ll_8 - cen_pt_0;
		float bl_1 = bb_7 + ll_9 - cen_pt_1;
		float tr_0 = tt_2 + rr_4 - cen_pt_0;
		float tr_1 = tt_3 + rr_5 - cen_pt_1;
		float br_0 = bb_6 + rr_4 - cen_pt_0;
		float br_1 = bb_7 + rr_5 - cen_pt_1;
		float pts_tr_0 = tr_0 * down_ratio / INPUT_W * img_width;
		float pts_br_0 = br_0 * down_ratio / INPUT_W * img_width;
		float pts_bl_0 = bl_0 * down_ratio / INPUT_W * img_width;
		float pts_tl_0 = tl_0 * down_ratio / INPUT_W * img_width;
		float pts_tr_1 = tr_1 * down_ratio / INPUT_H * img_height;
		float pts_br_1 = br_1 * down_ratio / INPUT_H * img_height;
		float pts_bl_1 = bl_1 * down_ratio / INPUT_H * img_height;
		float pts_tl_1 = tl_1 * down_ratio / INPUT_H * img_height;

		auto score = row[10];
		auto cls = row[11]; 
		
		if (score < CONF_THRESHOLD)//���Ŷ�ɸѡ
			continue;
		Detection box;
		box.conf = score;
		box.class_id = cls;
		float ploybox[8] = { pts_tr_0, pts_tr_1, pts_br_0, pts_br_1, pts_bl_0, pts_bl_1, pts_tl_0, pts_tl_1 };
		for (int i = 0; i < 8; i++) { 
			box.bbox[i] = ploybox[i];
		}

		if (m.count(box.class_id) == 0) {
			m.emplace(box.class_id, std::vector<Detection>());
		}
		m[box.class_id].push_back(box);
	}
	
	std::vector<Detection> res;//���ս��
	for (auto it = m.begin(); it != m.end(); it++) {//�ֱ𵼳�ÿһ������� ��ÿһ��������nms
		//std::cout << it->second[0].class_id << " --- " << std::endl;
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp);//�������Ŷȴ�С�Ӹߵ�������
		for (size_t m = 0; m < dets.size(); ++m) {
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n) {
				if (iou_poly(item.bbox, dets[n].bbox) > NMS_THRESHOLD) {//nmsɸѡ
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
	printf("��%d�����", res.size());
	auto end = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms,post process" << std::endl;
	
	std::cout << res.size() << std::endl;
	for (size_t j = 0; j < res.size(); j++) {
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Point> contours_item;

		float score = res[j].conf;
		float tl_0 = res[j].bbox[0];
		float tl_1 = res[j].bbox[1];
		float tr_0 = res[j].bbox[2];
		float tr_1 = res[j].bbox[3];
		float br_0 = res[j].bbox[4];
		float br_1 = res[j].bbox[5];
		float bl_0 = res[j].bbox[6];
		float bl_1 = res[j].bbox[7];

		contours_item.push_back(cv::Point(res[j].bbox[0], res[j].bbox[1]));
		contours_item.push_back(cv::Point(res[j].bbox[2], res[j].bbox[3]));
		contours_item.push_back(cv::Point(res[j].bbox[4], res[j].bbox[5]));
		contours_item.push_back(cv::Point(res[j].bbox[6], res[j].bbox[7]));
		contours.push_back(contours_item);

		//cv::drawContours(src, contours, -1, cv::Scalar(0, 255, 0), 1, 1);//Դ����Ļ�ͼ��ʽ����������б��
		// ���Լ��Ļ�ͼ���� ������С������ת��4������
		RotatedRect resultRect;
		resultRect = minAreaRect(contours_item);

		Point2f pt[4];
		resultRect.points(pt);
		line(src, pt[0], pt[1], Scalar(255, 0, 0), 2, 8);
		line(src, pt[1], pt[2], Scalar(255, 0, 0), 2, 8);
		line(src, pt[2], pt[3], Scalar(255, 0, 0), 2, 8);
		line(src, pt[3], pt[0], Scalar(255, 0, 0), 2, 8);
		string label = format("%.2f", res[j].conf);
		/*std::vector<string> class_names;
		class_names.push_back("bud");
		class_names.push_back("nobud");
		class_names.push_back("raw");

		label = class_names[res[j].class_id] + ":" + label;*/
		putText(src, label, pt[0], FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);

		//�ڲ���4���� ����ͨ����������жϷ���
		float tt_0 = (tl_0 + tr_0) / 2;
		float tt_1 = (tl_1 + tr_1) / 2;
		float rr_0 = (tr_0 + br_0) / 2;
		float rr_1 = (tr_1 + br_1) / 2;
		float bb_0 = (bl_0 + br_0) / 2;
		float bb_1 = (bl_1 + br_1) / 2;
		float ll_0 = (tl_0 + bl_0) / 2;
		float ll_1 = (tl_1 + bl_1) / 2;
		float cen_pts_0 = (tt_0 + rr_0 + bb_0 + ll_0) / 4;
		float cen_pts_1 = (tt_1 + rr_1 + bb_1 + ll_1) / 4;
		cv::line(src, cv::Point(int(cen_pts_0), int(cen_pts_1)), cv::Point(int(tt_0), int(tt_1)), (0, 0, 255), 1,
			1);
		cv::line(src, cv::Point(int(cen_pts_0), int(cen_pts_1)), cv::Point(int(rr_0), int(rr_1)), (255, 0, 255), 1,
			1);
		cv::line(src, cv::Point(int(cen_pts_0), int(cen_pts_1)), cv::Point(int(bb_0), int(bb_1)), (0, 255, 0), 1,
			1);
		cv::line(src, cv::Point(int(cen_pts_0), int(cen_pts_1)), cv::Point(int(ll_0), int(ll_1)), (255, 0, 0), 1,
			1);


	
		
	}
	cv::imshow("output.jpg", src);
	char c = cv::waitKey();
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

	system("pause");
    return 0;
}
