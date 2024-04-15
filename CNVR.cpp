#include "CNVR.h"

#include <cstdarg>

void StringAppendV(std::string* dst, const char* format, va_list ap) {
  // First try with a small fixed size buffer.
  static const int kFixedBufferSize = 1024;
  char fixed_buffer[kFixedBufferSize];

  // It is possible for methods that use a va_list to invalidate
  // the data in it upon use.  The fix is to make a copy
  // of the structure before using it and use that copy instead.
  va_list backup_ap;
  va_copy(backup_ap, ap);
  int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
  va_end(backup_ap);

  if (result < kFixedBufferSize) {
    if (result >= 0) {
      // Normal case - everything fits.
      dst->append(fixed_buffer, result);
      return;
    }

#ifdef _MSC_VER
    // Error or MSVC running out of space.  MSVC 8.0 and higher
    // can be asked about space needed with the special idiom below:
    va_copy(backup_ap, ap);
    result = vsnprintf(nullptr, 0, format, backup_ap);
    va_end(backup_ap);
#endif

    if (result < 0) {
      // Just an error.
      return;
    }
  }

  // Increase the buffer size to the size requested by vsnprintf,
  // plus one for the closing \0.
  const int variable_buffer_size = result + 1;
  std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

  // Restore the va_list before we use it again.
  va_copy(backup_ap, ap);
  result =
      vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
  va_end(backup_ap);

  if (result >= 0 && result < variable_buffer_size) {
    dst->append(variable_buffer.get(), result);
  }
}

std::string StringPrintf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  std::string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  return result;
}

void CudaSafeCall(const cudaError_t error, const std::string& file,
                  const int line) {
  if (error != cudaSuccess) {
    std::cerr << StringPrintf("%s in %s at line %i", cudaGetErrorString(error),
                              file.c_str(), line)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CudaCheckError(const char* file, const int line) {
  cudaError error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << StringPrintf("cudaCheckError() failed at %s:%i : %s", file,
                              line, cudaGetErrorString(error))
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  error = cudaDeviceSynchronize();
  if (cudaSuccess != error) {
    std::cerr << StringPrintf("cudaCheckError() with sync failed at %s:%i : %s",
                              file, line, cudaGetErrorString(error))
              << std::endl;
    std::cerr
        << "This error is likely caused by the graphics card timeout "
           "detection mechanism of your operating system. Please refer to "
           "the FAQ in the documentation on how to solve this problem."
        << std::endl;
    exit(EXIT_FAILURE);
  }
}

CNVR::CNVR() {}

CNVR::~CNVR()
{
    delete[] plane_hypotheses_host;
    delete[] costs_host;
    //delete[] normal_costs_host;

    for (int i = 0; i < num_images; ++i) {
        cudaDestroyTextureObject(texture_objects_host.images[i]);
        cudaFreeArray(cuArray[i]);
    }
    cudaFree(texture_objects_cuda);
    cudaFree(cameras_cuda);
    cudaFree(plane_hypotheses_cuda);
    cudaFree(pre_plane_hypotheses_cuda);
    cudaFree(costs_cuda);
    //cudaFree(normal_costs_cuda);
    cudaFree(pre_costs_cuda);
    cudaFree(rand_states_cuda);
    cudaFree(selected_views_cuda);
    cudaFree(depths_cuda);
    cudaFree(normals0_cuda);
    cudaFree(normals1_cuda);
    cudaFree(normals2_cuda);

    if (params.geom_consistency) {
        for (int i = 0; i < num_images; ++i) {
            cudaDestroyTextureObject(texture_depths_host.images[i]);
            cudaDestroyTextureObject(texture_normals0_host.images[i]);
            cudaDestroyTextureObject(texture_normals1_host.images[i]);
            cudaDestroyTextureObject(texture_normals2_host.images[i]);
            cudaFreeArray(cuDepthArray[i]);
            cudaFreeArray(cuNormal0Array[i]);
            cudaFreeArray(cuNormal1Array[i]);
            cudaFreeArray(cuNormal2Array[i]);
        }
        cudaFree(texture_depths_cuda);
        cudaFree(texture_normals0_cuda);
        cudaFree(texture_normals1_cuda);
        cudaFree(texture_normals2_cuda);
    }

    if (params.hierarchy) {
        delete[] scaled_plane_hypotheses_host;
        delete[] pre_costs_host;

        cudaFree(scaled_plane_hypotheses_cuda);
        cudaFree(pre_costs_cuda);
    }
    //std::cout << "CNVR destructor" << std::endl;
}

Camera ReadCamera(const std::string &cam_path)
{
    Camera camera;
    std::ifstream file(cam_path);

    std::string line;
    file >> line;

    for (int i = 0; i < 3; ++i) {
        file >> camera.R[3 * i + 0] >> camera.R[3 * i + 1] >> camera.R[3 * i + 2] >> camera.t[i];
    }

    float tmp[4];
    file >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
    file >> line;

    for (int i = 0; i < 3; ++i) {
        file >> camera.K[3 * i + 0] >> camera.K[3 * i + 1] >> camera.K[3 * i + 2];
    }

    float depth_num;
    float interval;
    file >> camera.depth_min >> interval >> depth_num >> camera.depth_max;

    return camera;
}

void  RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera)
{
    const int cols = depth.cols;
    const int rows = depth.rows;

    if (cols == src.cols && rows == src.rows) {
        dst = src.clone();
        return;
    }

    const float scale_x = cols / static_cast<float>(src.cols);
    const float scale_y = rows / static_cast<float>(src.rows);

    cv::resize(src, dst, cv::Size(cols,rows), 0, 0, cv::INTER_LINEAR);

    camera.K[0] *= scale_x;
    camera.K[2] *= scale_x;
    camera.K[4] *= scale_y;
    camera.K[5] *= scale_y;
    camera.width = cols;
    camera.height = rows;
}

float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera)
{
    float3 pointX;
    float3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    float3 C;
    C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}

void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

float GetAngle( const cv::Vec3f &v1, const cv::Vec3f &v2 )
{
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float angle = acosf(dot_product);
    //if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
    if ( angle != angle )
        return 0.0f;

    return angle;
}

int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth)
{
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage){
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t),1,inimage);
    fread(&h,sizeof(int32_t),1,inimage);
    fread(&w,sizeof(int32_t),1,inimage);
    fread(&nb,sizeof(int32_t),1,inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h*w*nb;
    depth = cv::Mat::zeros(h, w, CV_32F);
    fread(depth.data, sizeof(float), dataSize, inimage);
    fclose(inimage);
    return 0;
}

int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth)
{
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");
    if (!outimage) {
        std::cout << "Error opening file " << file_path << std::endl;
    }

    int32_t type = 1;
    int32_t h = depth.rows;
    int32_t w = depth.cols;
    int32_t nb = 1;

    fwrite(&type,sizeof(int32_t),1,outimage);
    fwrite(&h,sizeof(int32_t),1,outimage);
    fwrite(&w,sizeof(int32_t),1,outimage);
    fwrite(&nb,sizeof(int32_t),1,outimage);

    float* data = (float*)depth.data;

    int32_t datasize = w*h*nb;
    fwrite(data,sizeof(float),datasize,outimage);

    fclose(outimage);
    return 0;
}

int readNormalDmb (const std::string file_path, cv::Mat_<cv::Vec3f> &normal)
{
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage) {
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t),1,inimage);
    fread(&h,sizeof(int32_t),1,inimage);
    fread(&w,sizeof(int32_t),1,inimage);
    fread(&nb,sizeof(int32_t),1,inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h*w*nb;

    normal = cv::Mat::zeros(h, w, CV_32FC3);
    fread(normal.data, sizeof(float), dataSize, inimage);

    fclose(inimage);
    return 0;
}

int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal)
{
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");
    if (!outimage) {
        std::cout << "Error opening file " << file_path << std::endl;
    }

    int32_t type = 1; //float
    int32_t h = normal.rows;
    int32_t w = normal.cols;
    int32_t nb = 3;

    fwrite(&type,sizeof(int32_t),1,outimage);
    fwrite(&h,sizeof(int32_t),1,outimage);
    fwrite(&w,sizeof(int32_t),1,outimage);
    fwrite(&nb,sizeof(int32_t),1,outimage);

    float* data = (float*)normal.data;

    int32_t datasize = w*h*nb;
    fwrite(data,sizeof(float),datasize,outimage);

    fclose(outimage);
    return 0;
}

void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc)
{
    std::cout << "store 3D points to ply file" << std::endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath.c_str(), "wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size());
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    //fprintf(outputPly, "property float nx\n");
    //fprintf(outputPly, "property float ny\n");
    //fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    //write data
//#pragma omp parallel for
    for(size_t i = 0; i < pc.size(); i++) {
        const PointList &p = pc[i];
        float3 X = p.coord;
        //const float3 normal = p.normal;
        const float3 color = p.color;
        const char b_color = (int)color.x;
        const char g_color = (int)color.y;
        const char r_color = (int)color.z;

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
//#pragma omp critical
        {
            fwrite(&X.x,      sizeof(X.x), 1, outputPly);
            fwrite(&X.y,      sizeof(X.y), 1, outputPly);
            fwrite(&X.z,      sizeof(X.z), 1, outputPly);
            //fwrite(&normal.x, sizeof(normal.x), 1, outputPly);
            //fwrite(&normal.y, sizeof(normal.y), 1, outputPly);
            //fwrite(&normal.z, sizeof(normal.z), 1, outputPly);
            fwrite(&r_color,  sizeof(char), 1, outputPly);
            fwrite(&g_color,  sizeof(char), 1, outputPly);
            fwrite(&b_color,  sizeof(char), 1, outputPly);
        }

    }
    fclose(outputPly);
}

static float GetDisparity(const Camera &camera, const int2 &p, const float &depth)
{
    float point3D[3];
    point3D[0] = depth * (p.x - camera.K[2]) / camera.K[0];
    point3D[1] = depth * (p.y - camera.K[5]) / camera.K[4];
    point3D[3] = depth;

    return std::sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1] + point3D[2] * point3D[2]);
}

void CNVR::SetGeomConsistencyParams(bool multi_gemetry=false)
{
    params.geom_consistency = true;
    params.max_iterations = 2;
    if (multi_gemetry) {
       params.multi_geometry = true;
    }
}

void CNVR::SetHierarchyParams()
{
    params.hierarchy = true;
}

void CNVR::SetRepairParams() {
    params.repair = true;
}

void CNVR::SetNormalLambda(int iteration) {
    params.normal_lambda = 2*iteration;
}


void CNVR::InputInitialization(const std::string &dense_folder, const std::vector<Problem> &problems, const int idx)
{
    images.clear();
    cameras.clear();
    const Problem problem = problems[idx];

    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder = dense_folder + std::string("/cams");

    std::stringstream image_path;
    image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".jpg";
    cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat image_float;
    image_uint.convertTo(image_float, CV_32FC1);
    images.push_back(image_float);
    std::stringstream cam_path;
    cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << "_cam.txt";
    Camera camera = ReadCamera(cam_path.str());
    camera.height = image_float.rows;
    camera.width = image_float.cols;
    cameras.push_back(camera);

    size_t num_src_images = problem.src_image_ids.size();
    for (size_t i = 0; i < num_src_images; ++i) {
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i] << ".jpg";
        cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
        cv::Mat image_float;
        image_uint.convertTo(image_float, CV_32FC1);
        images.push_back(image_float);
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i] << "_cam.txt";
        Camera camera = ReadCamera(cam_path.str());
        camera.height = image_float.rows;
        camera.width = image_float.cols;
        cameras.push_back(camera);
    }

    // Scale cameras and images
    int max_image_size = problems[idx].cur_image_size;
    for (size_t i = 0; i < images.size(); ++i) {
        if (i > 0) {
            max_image_size = problems[problem.src_image_ids[i - 1]].cur_image_size;
        }

        if (images[i].cols <= max_image_size && images[i].rows <= max_image_size) {
            continue;
        }

        const float factor_x = static_cast<float>(max_image_size) / images[i].cols;
        const float factor_y = static_cast<float>(max_image_size) / images[i].rows;
        const float factor = factor_x<factor_y? factor_x: factor_y;

        const int new_cols = std::round(images[i].cols * factor);
        const int new_rows = std::round(images[i].rows * factor);

        const float scale_x = new_cols / static_cast<float>(images[i].cols);
        const float scale_y = new_rows / static_cast<float>(images[i].rows);

        cv::Mat_<float> scaled_image_float;
        cv::resize(images[i], scaled_image_float, cv::Size(new_cols,new_rows), 0, 0, cv::INTER_LINEAR);
        images[i] = scaled_image_float.clone();

        cameras[i].K[0] *= scale_x;
        cameras[i].K[2] *= scale_x;
        cameras[i].K[4] *= scale_y;
        cameras[i].K[5] *= scale_y;
        cameras[i].height = scaled_image_float.rows;
        cameras[i].width = scaled_image_float.cols;
    }
    params.depth_min = cameras[0].depth_min*0.6;
    params.depth_max = cameras[0].depth_max*1.4;
    std::cout << "depthe range: " << params.depth_min << " " << params.depth_max << std::endl;
    params.num_images = (int)images.size();
    std::cout << "num images: " << params.num_images << std::endl;
    params.disparity_min = cameras[0].K[0] * params.baseline / params.depth_max;
    params.disparity_max = cameras[0].K[0] * params.baseline / params.depth_min;

    if (params.geom_consistency) {
        depths.clear();
        normals0.clear();

        std::stringstream result_path;
        result_path << dense_folder << "/CNVR" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (params.multi_geometry) {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        cv::Mat_<float> ref_depth;
        readDepthDmb(depth_path, ref_depth);
        depths.push_back(ref_depth);
        for (size_t i = 0; i < num_src_images; ++i) {
            std::stringstream result_path;
            result_path << dense_folder << "/CNVR" << "/2333_" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i];
            std::string result_folder = result_path.str();
            std::string depth_path = result_folder + suffix;
            cv::Mat_<float> depth;
            readDepthDmb(depth_path, depth);
            depths.push_back(depth);
        }
        suffix = "/normals.dmb";
        if (params.multi_geometry) {
            suffix = "/normals_geom.dmb";
        }
        std::stringstream result_path_;
        result_path_<< dense_folder << "/CNVR" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder_ = result_path_.str();
        std::string normal_path = result_folder_ + suffix;
        cv::Mat_<cv::Vec3f> ref_normal;
        readNormalDmb(normal_path, ref_normal);
        std::vector<cv::Mat> channels;
        cv::split(ref_normal, channels);
        normals0.push_back(channels[0]);
        normals1.push_back(channels[1]);
        normals2.push_back(channels[2]);

        for (size_t i = 0; i < num_src_images; ++i) {
            std::stringstream result_path__;
            result_path__ << dense_folder << "/CNVR" << "/2333_" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i];
            std::string result_folder__ = result_path__.str();
            std::string normal_path = result_folder__ + suffix;
            cv::Mat_<cv::Vec3f> normal;
            readNormalDmb(normal_path, normal);
            std::vector<cv::Mat> channels_;
            cv::split(normal, channels_);
            normals0.push_back(channels_[0]);
            normals1.push_back(channels_[1]);
            normals2.push_back(channels_[2]);
        }

    }
}

void CNVR::CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem)
{
    num_images = (int)images.size();

    for (int i = 0; i < num_images; ++i) {
        int rows = images[i].rows;
        int cols = images[i].cols;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
        cudaMemcpy2DToArray (cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode  = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
    }
    cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
    cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cameras_cuda, sizeof(Camera) * (num_images));
    cudaMemcpy(cameras_cuda, &cameras[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);

    plane_hypotheses_host = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));
    cudaMalloc((void**)&pre_plane_hypotheses_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

    costs_host = new float[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&costs_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));
    cudaMalloc((void**)&pre_costs_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));

    cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (cameras[0].height * cameras[0].width));
    cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

    if (params.geom_consistency) {
        for (int i = 0; i < num_images; ++i) {
            int rows = depths[i].rows;
            int cols = depths[i].cols;

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuDepthArray[i], &channelDesc, cols, rows);
            cudaMemcpy2DToArray (cuDepthArray[i], 0, 0, depths[i].ptr<float>(), depths[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuDepthArray[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode  = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_depths_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaMalloc((void**)&texture_depths_cuda, sizeof(cudaTextureObjects));
        cudaMemcpy(texture_depths_cuda, &texture_depths_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

        //normal map dim0
        for (int i = 0; i < num_images; ++i) {
            int rows = normals0[i].rows;
            int cols = normals0[i].cols;


            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuNormal0Array[i], &channelDesc, cols, rows);
            cudaMemcpy2DToArray(cuNormal0Array[i], 0, 0, normals0[i].ptr<float>(), normals0[i].step[0], cols * sizeof(float), rows, cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuNormal0Array[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_normals0_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaMalloc((void**)&texture_normals0_cuda, sizeof(cudaTextureObjects));
        cudaMemcpy(texture_normals0_cuda, &texture_normals0_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

        //normal map dim1
        for (int i = 0; i < num_images; ++i) {
            int rows = normals1[i].rows;
            int cols = normals1[i].cols;


            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuNormal1Array[i], &channelDesc, cols, rows);
            cudaMemcpy2DToArray(cuNormal1Array[i], 0, 0, normals1[i].ptr<float>(), normals1[i].step[0], cols * sizeof(float), rows, cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuNormal1Array[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_normals1_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaMalloc((void**)&texture_normals1_cuda, sizeof(cudaTextureObjects));
        cudaMemcpy(texture_normals1_cuda, &texture_normals1_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

        //normal map dim2
        for (int i = 0; i < num_images; ++i) {
            int rows = normals2[i].rows;
            int cols = normals2[i].cols;


            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuNormal2Array[i], &channelDesc, cols, rows);
            cudaMemcpy2DToArray(cuNormal2Array[i], 0, 0, normals2[i].ptr<float>(), normals2[i].step[0], cols * sizeof(float), rows, cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuNormal2Array[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_normals2_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaMalloc((void**)&texture_normals2_cuda, sizeof(cudaTextureObjects));
        cudaMemcpy(texture_normals2_cuda, &texture_normals2_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);


        std::stringstream result_path;
        result_path << dense_folder << "/CNVR" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix_depth = "/depths.dmb";
        std::string suffix_normal = "/normals.dmb";
        if (params.multi_geometry) {
            suffix_depth = "/depths_geom.dmb";
            suffix_normal = "/normals_geom.dmb";
        }
        std::string depth_path = result_folder + suffix_depth;
        std::string normal_path = result_folder + suffix_normal;
        std::string cost_path = result_folder + "/costs.dmb";
        cv::Mat_<float> ref_depth;
        cv::Mat_<cv::Vec3f> ref_normal;
        cv::Mat_<float> ref_cost;
        cv::Mat_<float> ref_normal_cost;
        readDepthDmb(depth_path, ref_depth);
        depths.push_back(ref_depth);
        readNormalDmb(normal_path, ref_normal);
        readDepthDmb(cost_path, ref_cost);
        int width = ref_depth.cols;
        int height = ref_depth.rows;
        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int center = row * width + col;
                float4 plane_hypothesis;
                plane_hypothesis.x = ref_normal(row, col)[0];
                plane_hypothesis.y = ref_normal(row, col)[1];
                plane_hypothesis.z = ref_normal(row, col)[2];
                plane_hypothesis.w = ref_depth(row, col);
                plane_hypotheses_host[center] = plane_hypothesis;
                costs_host[center] = ref_cost(row, col);
            }
        }
        cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host, sizeof(float4) * width * height, cudaMemcpyHostToDevice);
        cudaMemcpy(costs_cuda, costs_host, sizeof(float) * width * height, cudaMemcpyHostToDevice);
        //cudaMemcpy(normal_costs_cuda, normal_costs_host, sizeof(float)* width* height, cudaMemcpyHostToDevice);
    }

    if (params.hierarchy) {
        std::stringstream result_path;
        result_path << dense_folder << "/CNVR" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string depth_path = result_folder + "/depths.dmb";
        std::string normal_path = result_folder + "/normals_geom.dmb";
        std::string cost_path = result_folder + "/costs.dmb";
        cv::Mat_<float> ref_depth;
        cv::Mat_<cv::Vec3f> ref_normal;
        cv::Mat_<float> ref_cost;
        readDepthDmb(depth_path, ref_depth);
        depths.push_back(ref_depth);
        readNormalDmb(normal_path, ref_normal);
        readDepthDmb(cost_path, ref_cost);
        int width = ref_normal.cols;
        int height = ref_normal.rows;
        scaled_plane_hypotheses_host= new float4[height * width];
        cudaMalloc((void**)&scaled_plane_hypotheses_cuda, sizeof(float4) * height * width);
        pre_costs_host = new float[height * width];
        cudaMalloc((void**)&pre_costs_cuda, sizeof(float) * cameras[0].height * cameras[0].width);
        if (width !=images[0].rows || height != images[0].cols) {
            params.upsample = true;
            params.scaled_cols = width;
            params.scaled_rows = height;
        }
        else {
            params.upsample = false;
        }
        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int center = row * width + col;
                float4 plane_hypothesis;
                plane_hypothesis.x = ref_normal(row, col)[0];
                plane_hypothesis.y = ref_normal(row, col)[1];
                plane_hypothesis.z = ref_normal(row, col)[2];
                if (params.upsample) {
                    plane_hypothesis.w = ref_cost(row, col);
                }
                else {
                    plane_hypothesis.w = ref_depth(row, col);
                }
                scaled_plane_hypotheses_host[center] = plane_hypothesis;
            }
        }

        for (int col = 0; col < cameras[0].width; ++col) {
            for (int row = 0; row < cameras[0].height; ++row) {
                int center = row * cameras[0].width + col;
                float4 plane_hypothesis;
                plane_hypothesis.w = ref_depth(row, col);
                plane_hypotheses_host[center] = plane_hypothesis;
            }
        }
        cudaMemcpy(scaled_plane_hypotheses_cuda, scaled_plane_hypotheses_host, sizeof(float4) * height * width, cudaMemcpyHostToDevice);
        cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host, sizeof(float4) * cameras[0].width * cameras[0].height, cudaMemcpyHostToDevice);
    }
}

int CNVR::GetReferenceImageWidth()
{
    return cameras[0].width;
}

int CNVR::GetReferenceImageHeight()
{
    return cameras[0].height;
}

cv::Mat CNVR::GetReferenceImage()
{
    return images[0];
}

float4 CNVR::GetPlaneHypothesis(const int index)
{
    return plane_hypotheses_host[index];
}

float CNVR::GetCost(const int index)
{
    return costs_host[index];
}


 void JBUAddImageToTextureFloatGray ( std::vector<cv::Mat_<float>>  &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[], const int &numSelViews)
{
    for (int i=0; i<numSelViews; i++) {
        int index = i;
        int rows = imgs[index].rows;
        int cols = imgs[index].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc (32, 0, 0,  0, cudaChannelFormatKindFloat);
        // Allocate array with correct size and number of channels
        cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
        cudaMemcpy2DToArray (cuArray[i], 0, 0, imgs[index].ptr<float>(), imgs[index].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL);
    }
    return;
}

 JBU::JBU(){}

 JBU::~JBU()
 {
     free(depth_h);

     cudaFree(depth_d);
     cudaFree(jp_d);
     cudaFree(jt_d);
 }

 void JBU::InitializeParameters(int n)
 {
     depth_h = (float*)malloc(sizeof(float) * n);

     cudaMalloc ((void**)&depth_d,  sizeof(float) * n);

     cudaMalloc((void**)&jp_d, sizeof(JBUParameters) * 1);
     cudaMemcpy(jp_d, &jp_h, sizeof(JBUParameters) * 1, cudaMemcpyHostToDevice);

     cudaMalloc((void**)&jt_d, sizeof(JBUTexObj) * 1);
     cudaMemcpy(jt_d, &jt_h, sizeof(JBUTexObj) * 1, cudaMemcpyHostToDevice);
     cudaDeviceSynchronize();
 }

void RunJBU(const cv::Mat_<float>  &scaled_image_float, const cv::Mat_<float> &src_depthmap, const std::string &dense_folder , const Problem &problem)
{
    uint32_t rows = scaled_image_float.rows;
    uint32_t cols = scaled_image_float.cols;
    int Imagescale = scaled_image_float.rows / src_depthmap.rows>scaled_image_float.cols / src_depthmap.cols? scaled_image_float.rows / src_depthmap.rows: scaled_image_float.cols / src_depthmap.cols;

    if (Imagescale == 1) {
        std::cout << "Image.rows = Depthmap.rows" << std::endl;
        return;
    }

    std::vector<cv::Mat_<float> > imgs(JBU_NUM);
    imgs[0] = scaled_image_float.clone();
    imgs[1] = src_depthmap.clone();

    JBU jbu;
    jbu.jp_h.height = rows;
    jbu.jp_h.width = cols;
    jbu.jp_h.s_height = src_depthmap.rows;
    jbu.jp_h.s_width = src_depthmap.cols;
    jbu.jp_h.Imagescale = Imagescale;
    JBUAddImageToTextureFloatGray(imgs, jbu.jt_h.imgs, jbu.cuArray, JBU_NUM);

    jbu.InitializeParameters(rows * cols);
    jbu.CudaRun();

    cv::Mat_<float> depthmap = cv::Mat::zeros( rows, cols, CV_32FC1 );

    for (uint32_t i = 0; i < cols; ++i) {
        for(uint32_t j = 0; j < rows; ++j) {
            int center = i + cols * j;
            if (jbu.depth_h[center] != jbu.depth_h[center]) {
                std::cout << "wrong!" << std::endl;
            }
            depthmap (j, i) = jbu.depth_h[center];
        }
    }

    cv::Mat_<float> disp0 = depthmap.clone();
    std::stringstream result_path;
#if defined(_WIN32)
    result_path << dense_folder << "\\CNVR" << "\\2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder =  result_path.str();
    std::string command = "mkdir " + result_folder;
    if(_access(result_folder.c_str(), 0) != 0){
        system(command.c_str());
    }
#else
    result_path << dense_folder << "/CNVR" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder =  result_path.str();
    mkdir(result_folder.c_str(), 777);
#endif

    std::string depth_path = result_folder + "/depths.dmb";
    writeDepthDmb ( depth_path, disp0 );

    for (int i=0; i < JBU_NUM; i++) {
        CUDA_SAFE_CALL( cudaDestroyTextureObject(jbu.jt_h.imgs[i]) );
        CUDA_SAFE_CALL( cudaFreeArray(jbu.cuArray[i]) );
    }
    cudaDeviceSynchronize();
}
