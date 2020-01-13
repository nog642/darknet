﻿#include "box.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.141592
#endif


box float_to_box(float const * const f)
{
    return (box){
        .x=f[0],
        .y=f[1],
        .w=f[2],
        .h=f[3]
    };
}


box float_to_box_stride(float const * const f, int const stride)
{
    return (box){
        .x=f[0],
        .y=f[1 * stride],
        .w=f[2 * stride],
        .h=f[3 * stride]
    };
}


dbox derivative(box const a, box const b)
{
    dbox d;
    d.dx = 0;
    d.dw = 0;
    d.dy = 0;
    d.dh = 0;
    d.dx = a.x < b.x ? 1.0 : -1.0;
    d.dy = a.y < b.y ? 1.0 : -1.0;
    d.dw = a.w < b.w ? 1.0 : -1.0;
    d.dh = a.h < b.h ? 1.0 : -1.0;
    return d;
}


/*
dbox derivative(box const a, box const b)
{
    dbox d;
    d.dx = 0;
    d.dw = 0;
    float const l1 = a.x - a.w / 2;
    float const l2 = b.x - b.w / 2;
    if (l1 > l2) {
        d.dx -= 1;
        d.dw += .5;
    }
    float const r1 = a.x + a.w/2;
    float const r2 = b.x + b.w/2;
    if (r1 < r2) {
        d.dx += 1;
        d.dw += .5;
    }
    if (l1 > r2) {
        d.dx = -1;
        d.dw = 0;
    }
    if (r1 < l2) {
        d.dx = 1;
        d.dw = 0;
    }

    d.dy = 0;
    d.dh = 0;
    float const t1 = a.y - a.h/2;
    float const t2 = b.y - b.h/2;
    if (t1 > t2) {
        d.dy -= 1;
        d.dh += .5;
    }
    float const b1 = a.y + a.h/2;
    float const b2 = b.y + b.h/2;
    if (b1 < b2) {
        d.dy += 1;
        d.dh += .5;
    }
    if (t1 > b2) {
        d.dy = -1;
        d.dh = 0;
    }
    if (b1 < t2) {
        d.dy = 1;
        d.dh = 0;
    }
    return d;
}
*/


// where c is the smallest box that fully encompasses a and b
boxabs box_c(box const a, box const b) {
    return (boxabs){
        .left=fmin(a.x - a.w / 2, b.x - b.w / 2),
        .right=fmax(a.x + a.w / 2, b.x + b.w / 2),
        .top=fmin(a.y - a.h / 2, b.y - b.h / 2),
        .bot=fmax(a.y + a.h / 2, b.y + b.h / 2)
    };
}


// representation from x, y, w, h to top, left, bottom, right
boxabs to_tblr(box const a) {
    return (boxabs){
        .left=a.x - (a.w / 2),
        .right=a.x + (a.w / 2),
        .top=a.y - (a.h / 2),
        .bot=a.y + (a.h / 2)
    };
}


float overlap(float const x1, float const w1, float const x2, float const w2)
{
    float const l1 = x1 - w1 / 2;
    float const l2 = x2 - w2 / 2;
    float const left = l1 > l2 ? l1 : l2;
    float const r1 = x1 + w1 / 2;
    float const r2 = x2 + w2 / 2;
    float const right = r1 < r2 ? r1 : r2;
    return right - left;
}


float box_intersection(box const a, box const b)
{
    float const w = overlap(a.x, a.w, b.x, b.w);
    float const h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) {
        return 0;
    }
    return w * h;
}


float box_union(box const a, box const b)
{
    return a.w * a.h + b.w * b.h - box_intersection(a, b);
}


float box_iou(box const a, box const b)
{
    // return box_intersection(a, b) / box_union(a, b);

    float const I = box_intersection(a, b);
    float const U = box_union(a, b);
    if (I == 0 || U == 0) {
        return 0;
    }
    return I / U;
}


float box_giou(box const a, box const b)
{
    boxabs const ba = box_c(a, b);
    float const w = ba.right - ba.left;
    float const h = ba.bot - ba.top;
    float const c = w * h;
    float const iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float const u = box_union(a, b);
    float const giou_term = (c - u) / c;
#ifdef DEBUG_PRINTS
    printf("  c: %f, u: %f, giou_term: %f\n", c, u, giou_term);
#endif
    return iou - giou_term;
}


// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
float box_diou(box a, box b)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w * w + h * h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float u = pow(d / c, 0.6);
    float diou_term = u;
#ifdef DEBUG_PRINTS
    printf("  c: %f, u: %f, riou_term: %f\n", c, u, diou_term);
#endif
    return iou - diou_term;
}

float box_diounms(box a, box b, float beta1)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w * w + h * h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float u = pow(d / c, beta1);
    float diou_term = u;
#ifdef DEBUG_PRINTS
    printf("  c: %f, u: %f, riou_term: %f\n", c, u, diou_term);
#endif
    return iou - diou_term;
}

// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
float box_ciou(box a, box b)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w * w + h * h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float u = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float d = u / c;
    float ar_gt = b.w / b.h;
    float ar_pred = a.w / a.h;
    float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
    float alpha = ar_loss / (1 - iou + ar_loss + 0.000001);
    float ciou_term = d + alpha * ar_loss;                   //ciou
#ifdef DEBUG_PRINTS
    printf("  c: %f, u: %f, riou_term: %f\n", c, u, ciou_term);
#endif
    return iou - ciou_term;
}


dxrep dx_box_iou(box const pred, box const truth, IOU_LOSS const iou_loss)
{
    boxabs const pred_tblr = to_tblr(pred);
    float const pred_t = fmin(pred_tblr.top, pred_tblr.bot);
    float const pred_b = fmax(pred_tblr.top, pred_tblr.bot);
    float const pred_l = fmin(pred_tblr.left, pred_tblr.right);
    float const pred_r = fmax(pred_tblr.left, pred_tblr.right);

    // dbox dover = derivative(pred, truth);
    // dbox diouu = diou(pred, truth);

    boxabs const truth_tblr = to_tblr(truth);

#ifdef DEBUG_PRINTS
    printf("\niou: %f, giou: %f\n", box_iou(pred, truth), box_giou(pred, truth));
    printf("pred: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n", pred.x, pred.y, pred.w, pred.h, pred_tblr.top, pred_tblr.bot, pred_tblr.left, pred_tblr.right);
    printf("truth: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n", truth.x, truth.y, truth.w, truth.h, truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
#endif
    // printf("pred (t,b,l,r): (%f, %f, %f, %f)\n", pred_t, pred_b, pred_l, pred_r);
    // printf("trut (t,b,l,r): (%f, %f, %f, %f)\n", truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
    dxrep ddx = {0};
    float const X = (pred_b - pred_t) * (pred_r - pred_l);
    float const Xhat = (truth_tblr.bot - truth_tblr.top) * (truth_tblr.right - truth_tblr.left);
    float const Ih = fmin(pred_b, truth_tblr.bot) - fmax(pred_t, truth_tblr.top);
    float const Iw = fmin(pred_r, truth_tblr.right) - fmax(pred_l, truth_tblr.left);
    float const I = Iw * Ih;
    float const U = X + Xhat - I;

    float S = (pred.x-truth.x)*(pred.x-truth.x)+(pred.y-truth.y)*(pred.y-truth.y);
    float giou_Cw = fmax(pred_r, truth_tblr.right) - fmin(pred_l, truth_tblr.left);
    float giou_Ch = fmax(pred_b, truth_tblr.bot) - fmin(pred_t, truth_tblr.top);
    float giou_C = giou_Cw * giou_Ch;
    //float IoU = I / U;
//#ifdef DEBUG_PRINTS
    //printf("X: %f", X);
    //printf(", Xhat: %f", Xhat);
    //printf(", Ih: %f", Ih);
    //printf(", Iw: %f", Iw);
    //printf(", I: %f", I);
    //printf(", U: %f", U);
    //printf(", IoU: %f\n", I / U);
//#endif

    // Partial Derivatives, derivatives
    float const dX_wrt_t = -1 * (pred_r - pred_l);
    float const dX_wrt_b = pred_r - pred_l;
    float const dX_wrt_l = -1 * (pred_b - pred_t);
    float const dX_wrt_r = pred_b - pred_t;
    // UNUSED
    // // Ground truth
    // float dXhat_wrt_t = -1 * (truth_tblr.right - truth_tblr.left);
    // float dXhat_wrt_b = truth_tblr.right - truth_tblr.left;
    // float dXhat_wrt_l = -1 * (truth_tblr.bot - truth_tblr.top);
    // float dXhat_wrt_r = truth_tblr.bot - truth_tblr.top;

    // gradient of I min/max in IoU calc (prediction)
    float const dI_wrt_t = pred_t > truth_tblr.top ? (-1 * Iw) : 0;
    float const dI_wrt_b = pred_b < truth_tblr.bot ? Iw : 0;
    float const dI_wrt_l = pred_l > truth_tblr.left ? (-1 * Ih) : 0;
    float const dI_wrt_r = pred_r < truth_tblr.right ? Ih : 0;
    // derivative of U with regard to x
    float const dU_wrt_t = dX_wrt_t - dI_wrt_t;
    float const dU_wrt_b = dX_wrt_b - dI_wrt_b;
    float const dU_wrt_l = dX_wrt_l - dI_wrt_l;
    float const dU_wrt_r = dX_wrt_r - dI_wrt_r;
    // gradient of C min/max in IoU calc (prediction)
    float const dC_wrt_t = pred_t < truth_tblr.top ? (-1 * giou_Cw) : 0;
    float const dC_wrt_b = pred_b > truth_tblr.bot ? giou_Cw : 0;
    float const dC_wrt_l = pred_l < truth_tblr.left ? (-1 * giou_Ch) : 0;
    float const dC_wrt_r = pred_r > truth_tblr.right ? giou_Ch : 0;

    float p_dt = 0;
    float p_db = 0;
    float p_dl = 0;
    float p_dr = 0;
    if (U > 0) {
        p_dt = (U * dI_wrt_t - I * dU_wrt_t) / (U * U);
        p_db = (U * dI_wrt_b - I * dU_wrt_b) / (U * U);
        p_dl = (U * dI_wrt_l - I * dU_wrt_l) / (U * U);
        p_dr = (U * dI_wrt_r - I * dU_wrt_r) / (U * U);
    }
    // apply grad from prediction min/max for correct corner selection
    p_dt = pred_tblr.top < pred_tblr.bot ? p_dt : p_db;
    p_db = pred_tblr.top < pred_tblr.bot ? p_db : p_dt;
    p_dl = pred_tblr.left < pred_tblr.right ? p_dl : p_dr;
    p_dr = pred_tblr.left < pred_tblr.right ? p_dr : p_dl;

    if (iou_loss == GIOU) {
        if (giou_C > 0) {
            // apply "C" term from gIOU
            p_dt += ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
            p_db += ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
            p_dl += ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
            p_dr += ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
        }
        if (Iw <= 0 || Ih <= 0) {
            p_dt = ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
            p_db = ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
            p_dl = ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
            p_dr = ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
        }
    }

    float Ct = fmin(pred.y - pred.h / 2,truth.y - truth.h / 2);
    float Cb = fmax(pred.y + pred.h / 2,truth.y + truth.h / 2);
    float Cl = fmin(pred.x - pred.w / 2,truth.x - truth.w / 2);
    float Cr = fmax(pred.x + pred.w / 2,truth.x + truth.w / 2);
    float Cw = Cr - Cl;
    float Ch = Cb - Ct;
    float C = Cw * Cw + Ch * Ch;

    float dCt_dx = 0;
    float dCt_dy = pred_t < truth_tblr.top ? 1 : 0;
    float dCt_dw = 0;
    float dCt_dh = pred_t < truth_tblr.top ? -0.5 : 0;

    float dCb_dx = 0;
    float dCb_dy = pred_b > truth_tblr.bot ? 1 : 0;
    float dCb_dw = 0;
    float dCb_dh = pred_b > truth_tblr.bot ? 0.5: 0;

    float dCl_dx = pred_l < truth_tblr.left ? 1 : 0;
    float dCl_dy = 0;
    float dCl_dw = pred_l < truth_tblr.left ? -0.5 : 0;
    float dCl_dh = 0;

    float dCr_dx = pred_r > truth_tblr.right ? 1 : 0;
    float dCr_dy = 0;
    float dCr_dw = pred_r > truth_tblr.right ? 0.5 : 0;
    float dCr_dh = 0;

    float dCw_dx = dCr_dx - dCl_dx;
    float dCw_dy = dCr_dy - dCl_dy;
    float dCw_dw = dCr_dw - dCl_dw;
    float dCw_dh = dCr_dh - dCl_dh;

    float dCh_dx = dCb_dx - dCt_dx;
    float dCh_dy = dCb_dy - dCt_dy;
    float dCh_dw = dCb_dw - dCt_dw;
    float dCh_dh = dCb_dh - dCt_dh;

    // UNUSED
    //// ground truth
    //float dI_wrt_xhat_t = pred_t < truth_tblr.top ? (-1 * Iw) : 0;
    //float dI_wrt_xhat_b = pred_b > truth_tblr.bot ? Iw : 0;
    //float dI_wrt_xhat_l = pred_l < truth_tblr.left ? (-1 * Ih) : 0;
    //float dI_wrt_xhat_r = pred_r > truth_tblr.right ? Ih : 0;

    // Final IOU loss (prediction) (negative of IOU gradient, we want the negative loss)
    float p_dx = 0;
    float p_dy = 0;
    float p_dw = 0;
    float p_dh = 0;

    p_dx = p_dl + p_dr;           //p_dx, p_dy, p_dw and p_dh are the gradient of IoU or GIoU.
    p_dy = p_dt + p_db;
    p_dw = (p_dr - p_dl);         //For dw and dh, we do not divided by 2.
    p_dh = (p_db - p_dt);

    // https://github.com/Zzh-tju/DIoU-darknet
    // https://arxiv.org/abs/1911.08287
    if (iou_loss == DIOU) {
        if (C > 0) {
            p_dx += (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
            p_dy += (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
            p_dw += (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C);
            p_dh += (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C);
        }
	if (Iw <= 0 || Ih <= 0) {
            p_dx = (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
            p_dy = (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
            p_dw = (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C);
            p_dh = (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C);
        }
    }
	// The following codes are calculating the gradient of ciou.

    if (iou_loss == CIOU) {
	    float ar_gt = truth.w / truth.h;
        float ar_pred = pred.w / pred.h;
        float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
	    float alpha = ar_loss / (1 - I/U + ar_loss + 0.000001);
	    float ar_dw=8/(M_PI*M_PI)*(atan(ar_gt)-atan(ar_pred))*pred.h;
        float ar_dh=-8/(M_PI*M_PI)*(atan(ar_gt)-atan(ar_pred))*pred.w;
        if (C > 0) {
            p_dx += (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
            p_dy += (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
            p_dw += (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C) + alpha * ar_dw;
            p_dh += (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C) + alpha * ar_dh;
        }
	    if (Iw <= 0 || Ih <= 0) {
            p_dx = (2*(truth.x-pred.x)*C-(2*Cw*dCw_dx+2*Ch*dCh_dx)*S) / (C * C);
            p_dy = (2*(truth.y-pred.y)*C-(2*Cw*dCw_dy+2*Ch*dCh_dy)*S) / (C * C);
            p_dw = (2*Cw*dCw_dw+2*Ch*dCh_dw)*S / (C * C) + alpha * ar_dw;
            p_dh = (2*Cw*dCw_dh+2*Ch*dCh_dh)*S / (C * C) + alpha * ar_dh;
        }
    }

    ddx.dt = p_dx;      //We follow the original code released from GDarknet. So in yolo_layer.c, dt, db, dl, dr are already dx, dy, dw, dh.
    ddx.db = p_dy;
    ddx.dl = p_dw;
    ddx.dr = p_dh;

    // UNUSED
    //// ground truth
    //float gt_dt = ((U * dI_wrt_xhat_t) - (I * (dXhat_wrt_t - dI_wrt_xhat_t))) / (U * U);
    //float gt_db = ((U * dI_wrt_xhat_b) - (I * (dXhat_wrt_b - dI_wrt_xhat_b))) / (U * U);
    //float gt_dl = ((U * dI_wrt_xhat_l) - (I * (dXhat_wrt_l - dI_wrt_xhat_l))) / (U * U);
    //float gt_dr = ((U * dI_wrt_xhat_r) - (I * (dXhat_wrt_r - dI_wrt_xhat_r))) / (U * U);

    // no min/max grad applied
    //dx.dt = dt;
    //dx.db = db;
    //dx.dl = dl;
    //dx.dr = dr;

    //// sum in gt -- THIS DOESNT WORK
    //dx.dt += gt_dt;
    //dx.db += gt_db;
    //dx.dl += gt_dl;
    //dx.dr += gt_dr;

    //// instead, look at the change between pred and gt, and weight t/b/l/r appropriately...
    //// need the real derivative here (I think?)
    //float delta_t = fmax(truth_tblr.top, pred_t) - fmin(truth_tblr.top, pred_t);
    //float delta_b = fmax(truth_tblr.bot, pred_b) - fmin(truth_tblr.bot, pred_b);
    //float delta_l = fmax(truth_tblr.left, pred_l) - fmin(truth_tblr.left, pred_l);
    //float delta_r = fmax(truth_tblr.right, pred_r) - fmin(truth_tblr.right, pred_r);

    //dx.dt *= delta_t / (delta_t + delta_b);
    //dx.db *= delta_b / (delta_t + delta_b);
    //dx.dl *= delta_l / (delta_l + delta_r);
    //dx.dr *= delta_r / (delta_l + delta_r);

    // UNUSED
    //// ground truth
    //float gt_dt = ((U * dI_wrt_xhat_t) - (I * (dXhat_wrt_t - dI_wrt_xhat_t))) / (U * U);
    //float gt_db = ((U * dI_wrt_xhat_b) - (I * (dXhat_wrt_b - dI_wrt_xhat_b))) / (U * U);
    //float gt_dl = ((U * dI_wrt_xhat_l) - (I * (dXhat_wrt_l - dI_wrt_xhat_l))) / (U * U);
    //float gt_dr = ((U * dI_wrt_xhat_r) - (I * (dXhat_wrt_r - dI_wrt_xhat_r))) / (U * U);

    // no min/max grad applied
    //dx.dt = dt;
    //dx.db = db;
    //dx.dl = dl;
    //dx.dr = dr;

    // apply grad from prediction min/max for correct corner selection
    //dx.dt = pred_tblr.top < pred_tblr.bot ? p_dt : p_db;
    //dx.db = pred_tblr.top < pred_tblr.bot ? p_db : p_dt;
    //dx.dl = pred_tblr.left < pred_tblr.right ? p_dl : p_dr;
    //dx.dr = pred_tblr.left < pred_tblr.right ? p_dr : p_dl;

    //// sum in gt -- THIS DOESNT WORK
    //dx.dt += gt_dt;
    //dx.db += gt_db;
    //dx.dl += gt_dl;
    //dx.dr += gt_dr;

    //// instead, look at the change between pred and gt, and weight t/b/l/r appropriately...
    //// need the real derivative here (I think?)
    //float delta_t = fmax(truth_tblr.top, pred_t) - fmin(truth_tblr.top, pred_t);
    //float delta_b = fmax(truth_tblr.bot, pred_b) - fmin(truth_tblr.bot, pred_b);
    //float delta_l = fmax(truth_tblr.left, pred_l) - fmin(truth_tblr.left, pred_l);
    //float delta_r = fmax(truth_tblr.right, pred_r) - fmin(truth_tblr.right, pred_r);

    //dx.dt *= delta_t / (delta_t + delta_b);
    //dx.db *= delta_b / (delta_t + delta_b);
    //dx.dl *= delta_l / (delta_l + delta_r);
    //dx.dr *= delta_r / (delta_l + delta_r);

//#ifdef DEBUG_PRINTS
    /*printf("  directions dt: ");
    if ((pred_tblr.top < truth_tblr.top && dx.dt > 0) || (pred_tblr.top > truth_tblr.top && dx.dt < 0)) {
      printf("✓");
    } else {
      printf("𝒙");
    }
    printf(", ");
    if ((pred_tblr.bot < truth_tblr.bot && dx.db > 0) || (pred_tblr.bot > truth_tblr.bot && dx.db < 0)) {
      printf("✓");
    } else {
      printf("𝒙");
    }
    printf(", ");
    if ((pred_tblr.left < truth_tblr.left && dx.dl > 0) || (pred_tblr.left > truth_tblr.left && dx.dl < 0)) {
      printf("✓");
    } else {
      printf("𝒙");
    }
    printf(", ");
    if ((pred_tblr.right < truth_tblr.right && dx.dr > 0) || (pred_tblr.right > truth_tblr.right && dx.dr < 0)) {
      printf("✓");
    } else {
      printf("𝒙");
    }
    printf("\n");

    printf("dx dt:%f", dx.dt);
    printf(", db: %f", dx.db);
    printf(", dl: %f", dx.dl);
    printf(", dr: %f | ", dx.dr);
#endif

#ifdef DEBUG_NAN
    if (isnan(dx.dt)) { printf("dt isnan\n"); }
    if (isnan(dx.db)) { printf("db isnan\n"); }
    if (isnan(dx.dl)) { printf("dl isnan\n"); }
    if (isnan(dx.dr)) { printf("dr isnan\n"); }
#endif

//    // No update if 0 or nan
//    if (dx.dt == 0 || isnan(dx.dt)) { dx.dt = 1; }
//    if (dx.db == 0 || isnan(dx.db)) { dx.db = 1; }
//    if (dx.dl == 0 || isnan(dx.dl)) { dx.dl = 1; }
//    if (dx.dr == 0 || isnan(dx.dr)) { dx.dr = 1; }
//
//#ifdef DEBUG_PRINTS
//    printf("dx dt:%f (t: %f, p: %f)", dx.dt, gt_dt, p_dt);
//    printf(", db: %f (t: %f, p: %f)", dx.db, gt_db, p_db);
//    printf(", dl: %f (t: %f, p: %f)", dx.dl, gt_dl, p_dl);
//    printf(", dr: %f (t: %f, p: %f) | ", dx.dr, gt_dr, p_dr);
//#endif */
    return ddx;
}


float box_rmse(box const a, box const b)
{
    return sqrt(pow(a.x - b.x, 2) +
                pow(a.y - b.y, 2) +
                pow(a.w - b.w, 2) +
                pow(a.h - b.h, 2));
}


dbox dintersect(box const a, box const b)
{
    float const w = overlap(a.x, a.w, b.x, b.w);
    float const h = overlap(a.y, a.h, b.y, b.h);
    dbox const dover = derivative(a, b);

    return (dbox){
        .dx=dover.dx * h,
        .dy=dover.dy * w,
        .dw=dover.dw * h,
        .dh=dover.dh * w
    };
}


dbox dunion(box const a, box const b)
{
    dbox const di = dintersect(a, b);
    return (dbox){
        .dx=-di.dx,
        .dy=-di.dy,
        .dw=a.h - di.dw,
        .dh=a.w - di.dh
    };
}


void test_dunion()
{
    box const a = {0, 0, 1, 1};
    box const dxa = {0 + .0001, 0, 1, 1};
    box const dya = {0, 0 + .0001, 1, 1};
    box const dwa = {0, 0, 1 + .0001, 1};
    box const dha = {0, 0, 1, 1 + .0001};

    box const b = {.5, .5, .2, .2};
    dbox const di = dunion(a, b);
    printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float const inter = box_union(a, b);
    float xinter = box_union(dxa, b);
    float yinter = box_union(dya, b);
    float winter = box_union(dwa, b);
    float hinter = box_union(dha, b);
    xinter = (xinter - inter) / .0001;
    yinter = (yinter - inter) / .0001;
    winter = (winter - inter) / .0001;
    hinter = (hinter - inter) / .0001;
    printf("Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}


void test_dintersect()
{
    box const a = {0, 0, 1, 1};
    box const dxa = {0 + .0001, 0, 1, 1};
    box const dya = {0, 0 + .0001, 1, 1};
    box const dwa = {0, 0, 1 + .0001, 1};
    box const dha = {0, 0, 1, 1 + .0001};

    box const b = {.5, .5, .2, .2};
    dbox const di = dintersect(a, b);
    printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float const inter =  box_intersection(a, b);
    float xinter = box_intersection(dxa, b);
    float yinter = box_intersection(dya, b);
    float winter = box_intersection(dwa, b);
    float hinter = box_intersection(dha, b);
    xinter = (xinter - inter) / .0001;
    yinter = (yinter - inter) / .0001;
    winter = (winter - inter) / .0001;
    hinter = (hinter - inter) / .0001;
    printf("Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}


void test_box()
{
    test_dintersect();
    test_dunion();
    box const a = {0, 0, 1, 1};
    box const dxa = {0 + .00001, 0, 1, 1};
    box const dya = {0, 0 + .00001, 1, 1};
    box const dwa = {0, 0, 1 + .00001, 1};
    box const dha = {0, 0, 1, 1 + .00001};

    box const b = {.5, 0, .2, .2};

    float iou = box_iou(a,b);
    iou = (1 - iou) * (1 - iou);
    printf("%f\n", iou);
    dbox const d = diou(a, b);
    printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

    float xiou = box_iou(dxa, b);
    float yiou = box_iou(dya, b);
    float wiou = box_iou(dwa, b);
    float hiou = box_iou(dha, b);
    xiou = ((1 - xiou) * (1 - xiou) - iou) / (.00001);
    yiou = ((1 - yiou) * (1 - yiou) - iou) / (.00001);
    wiou = ((1 - wiou) * (1 - wiou) - iou) / (.00001);
    hiou = ((1 - hiou) * (1 - hiou) - iou) / (.00001);
    printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
}


dbox diou(box const a, box const b)
{
    float const u = box_union(a, b);
    float const i = box_intersection(a, b);
    dbox const di = dintersect(a, b);
    dbox const du = dunion(a, b);

    dbox dd = {0, 0, 0, 0};

    if (i <= 0 || 1) {
        dd.dx = b.x - a.x;
        dd.dy = b.y - a.y;
        dd.dw = b.w - a.w;
        dd.dh = b.h - a.h;
        return dd;
    }

    dd.dx = (di.dx*u - du.dx*i) / (u*u);
    dd.dy = (di.dy*u - du.dy*i) / (u*u);
    dd.dw = (di.dw*u - du.dw*i) / (u*u);
    dd.dh = (di.dh*u - du.dh*i) / (u*u);
    return dd;
}


typedef struct {
    int index;
    int class_id;
    float * * probs;
} sortable_bbox;


int nms_comparator(void const * const pa, void const * const pb)
{
    sortable_bbox const a = *(sortable_bbox *)pa;
    sortable_bbox const b = *(sortable_bbox *)pb;
    float const diff = a.probs[a.index][b.class_id] - b.probs[b.index][b.class_id];
    if (diff < 0) {
        return 1;
    }
    if (diff > 0) {
        return -1;
    }
    return 0;
}


void do_nms_sort_v2(box const * const boxes, float * * const probs, int const total, int const classes, float const thresh)
{
    sortable_bbox * s = malloc(total * sizeof(sortable_bbox));
    for (int i = 0; i < total; ++i) {
        s[i].index = i;
        s[i].class_id = 0;
        s[i].probs = probs;
    }

    for (int k = 0; k < classes; ++k) {
        for (int i = 0; i < total; ++i) {
            s[i].class_id = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for (int i = 0; i < total; ++i) {
            if (probs[s[i].index][k] == 0) {
                continue;
            }
            box const a = boxes[s[i].index];
            for (int j = i + 1; j < total; ++j) {
                box const b = boxes[s[j].index];
                if (box_iou(a, b) > thresh) {
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }

    free(s);
}


int nms_comparator_v3(const void* pa, const void* pb)
{
    detection a = *(detection*)pa;
    detection b = *(detection*)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class]; // there is already: prob = objectness*prob
    }
    else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) {
        return 1;
    } else if (diff > 0) {
        return -1;
    }
    return 0;
}


void do_nms_obj(detection* dets, int total, int classes, float thresh)
{
    int i;
    int j;
    int k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (i = 0; i < total; ++i) {
        dets[i].sort_class = -1;
    }

    qsort(dets, total, sizeof(detection), nms_comparator_v3);
    for (i = 0; i < total; ++i) {
        if (dets[i].objectness == 0) continue;
        box a = dets[i].bbox;
        for (j = i + 1; j < total; ++j) {
            if (dets[j].objectness == 0) continue;
            box b = dets[j].bbox;
            if (box_iou(a, b) > thresh) {
                dets[j].objectness = 0;
                for (k = 0; k < classes; ++k) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}


void do_nms_sort(detection* dets, int total, int classes, float thresh)
{
    int i;
    int j;
    int k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator_v3);
        for (i = 0; i < total; ++i) {
            // printf("  k = %d, \t i = %d \n", k, i);
            if (dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}


void do_nms(box* boxes, float** probs, int total, int classes, float thresh)
{
    int i;
    int j;
    int k;
    for (i = 0; i < total; ++i) {
        int any = 0;
        for (k = 0; k < classes; ++k) {
            any = any || (probs[i][k] > 0);
        }
        if (!any) {
            continue;
        }
        for (j = i + 1; j < total; ++j) {
            if (box_iou(boxes[i], boxes[j]) > thresh) {
                for (k = 0; k < classes; ++k) {
                    if (probs[i][k] < probs[j][k]) {
                        probs[i][k] = 0;
                    } else {
                        probs[j][k] = 0;
                    }
                }
            }
        }
    }
}


// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
void diounms_sort(detection *dets, int total, int classes, float thresh, NMS_KIND nms_kind, float beta1)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator_v3);
        for (i = 0; i < total; ++i)
        {
            if (dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh && nms_kind == CORNERS_NMS) {
                    // float sum_prob = pow(dets[i].prob[k], 2) + pow(dets[j].prob[k], 2);
                    // float alpha_prob = pow(dets[i].prob[k], 2) / sum_prob;
                    // float beta_prob = pow(dets[j].prob[k], 2) / sum_prob;
                    // dets[i].bbox.x = (dets[i].bbox.x * alpha_prob + dets[j].bbox.x * beta_prob);
                    // dets[i].bbox.y = (dets[i].bbox.y * alpha_prob + dets[j].bbox.y * beta_prob);
                    // dets[i].bbox.w = (dets[i].bbox.w * alpha_prob + dets[j].bbox.w * beta_prob);
                    // dets[i].bbox.h = (dets[i].bbox.h * alpha_prob + dets[j].bbox.h * beta_prob);
                    // if (dets[j].points == YOLO_CENTER && (dets[i].points & dets[j].points) == 0) {
                    //     dets[i].bbox.x = (dets[i].bbox.x * alpha_prob + dets[j].bbox.x * beta_prob);
                    //     dets[i].bbox.y = (dets[i].bbox.y * alpha_prob + dets[j].bbox.y * beta_prob);
                    // } else if ((dets[i].points & dets[j].points) == 0) {
                    //     dets[i].bbox.w = (dets[i].bbox.w * alpha_prob + dets[j].bbox.w * beta_prob);
                    //     dets[i].bbox.h = (dets[i].bbox.h * alpha_prob + dets[j].bbox.h * beta_prob);
                    // }
                    // dets[i].points |= dets[j].points;
                    dets[j].prob[k] = 0;
                } else if (box_iou(a, b) > thresh && nms_kind == GREEDY_NMS) {
                    dets[j].prob[k] = 0;
                } else {
                    if (box_diounms(a, b, beta1) > thresh && nms_kind == DIOU_NMS) {
                        dets[j].prob[k] = 0;
                    }
                }
            }

            //if ((nms_kind == CORNERS_NMS) && (dets[i].points != (YOLO_CENTER | YOLO_LEFT_TOP | YOLO_RIGHT_BOTTOM)))
            //    dets[i].prob[k] = 0;
        }
    }
}


box encode_box(box b, box anchor)
{
    box encode;
    encode.x = (b.x - anchor.x) / anchor.w;
    encode.y = (b.y - anchor.y) / anchor.h;
    encode.w = log2(b.w / anchor.w);
    encode.h = log2(b.h / anchor.h);
    return encode;
}


box decode_box(box b, box anchor)
{
    box decode;
    decode.x = b.x * anchor.w + anchor.x;
    decode.y = b.y * anchor.h + anchor.y;
    decode.w = pow(2., b.w) * anchor.w;
    decode.h = pow(2., b.h) * anchor.h;
    return decode;
}
