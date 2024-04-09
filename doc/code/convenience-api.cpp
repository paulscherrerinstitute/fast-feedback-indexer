#include <Eigen/LU>
#include <iostream>
#include <cmath>
#include <atomic>
#include <thread>
#include <chrono>
#include <ffbidx/refine.h>

using std::cout;
using std::flush;
using std::atomic_bool;
using Mx3 = Eigen::MatrixX3f;
using M3 = Eigen::Matrix3f;
using Vx = Eigen::VectorXf;
using Eigen::Map;
using LU = Eigen::FullPivLU<M3>;
using namespace std::chrono_literals;
using namespace fast_feedback;
using ifss = refine::indexer_ifss<>;
using refine::best_cell;
constexpr float PI = 3.14159265358979323846;
constexpr float _60 = PI/3.0f;

void callback(void *data) {
    cout << '!' << flush;
    ((atomic_bool*)data)->store(true);
};

int main(int argc, char* argv[]) {
    float coords[3][10] = {
        {1.0, 4.0, 1.0, 2.0, 6.0, 4.0, 7.0, 8.0, 5.0, 9.0},
        {2.0, 5.0, 1.0, 1.0, 3.0, 5.0, 1.0, 2.0, 7.0, 8.0},
        {3.0, 6.0, 7.0, 4.0, 1.0, 2.0, 3.0, 6.0, 5.0, 8.0}
    };
    float base[3][3] = {
        {2.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    float rot60z[3][3] = {
        {  cosf(_60), sinf(_60), 0 },
        { -sinf(_60), cosf(_60), 0 },
        {          0,         0, 1 }
    };
    Mx3 Coords = Map<Mx3>((float*)coords, 10, 3);
    M3 B0 = Map<M3>((float*)base, 3, 3);
    M3 R60z = Map<M3>((float*)rot60z, 3, 3);
    M3 B = B0 * R60z.transpose();
    LU lu(B);
    M3 Binv = lu.inverse();
    Mx3 Spots = Coords * Binv.transpose();

    cout << "True Coords:\n" << Coords
         << "\nB0:\n" << B0
         << "\nR60z:\n" << R60z
         << "\nB:\n" << B
         << "\nBinv:\n" << Binv
         << "\nSpots:\n" << Spots << '\n';

    config_persistent<> cp{
        32, 1, 10, 32, true
    };
    config_runtime<> cr{};
    refine::config_ifss<> ci{};

    ifss indexer(cp, cr, ci);
    indexer.spotM() = Spots;
    indexer.iCellM() = B0;

    atomic_bool finished{false};
    indexer.index_start(1, 10, callback, &finished);
    while (! finished.load()) {
        cout << '.';
        std::this_thread::sleep_for(100us);
    }
    cout << '\n';
    indexer.index_end();
    unsigned best = best_cell(indexer.oScoreV());

    cout << "Cand:\n" << indexer.oCellM()
         << "\nScore:\n" << indexer.oScoreV()
         << "\nBest: " << best << '\n';

    float sf = indexer.oScore(best);
    M3 Bf = indexer.oCell(best);
    bool viable = (sf < .001f);

    cout << "B found:\n" << Bf
         << "\nScore: " << sf << ", Viable: " << (viable?"yes":"no")
         << "\nComputed Coords:\n" << (Spots * Bf.transpose()) << '\n';
}
