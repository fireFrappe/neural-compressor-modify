// Copyright (c) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import { Component, Inject, OnInit } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-optimization-form',
  templateUrl: './optimization-form.component.html',
  styleUrls: ['./optimization-form.component.scss', './../error/error.component.scss', './../project-form/project-form.component.scss',]
})
export class OptimizationFormComponent implements OnInit {

  precisions = [];
  precisionsPyTorch = [];
  precisionsOther = [];
  precisionId: number;

  optimizationTypes = [];
  optimizationTypeId: number;

  datasets = [];
  datasetId = 0;
  name: string;

  constructor(
    public modelService: ModelService,
    @Inject(MAT_DIALOG_DATA) public data
  ) { }

  ngOnInit(): void {
    this.name = 'Optimization' + String(this.data.index + 1);
    this.getPrecisions();
    this.getDatasets();
    this.modelService.datasetCreated$.subscribe(response => this.getDatasets());
  }

  getPrecisions() {
    this.modelService.getDictionary('precisions')
      .subscribe(
        (response: { precisions: any }) => {
          this.precisions = response.precisions;
          this.precisionId = this.precisions.find(x => x.name === 'int8').id;
          this.precisions.forEach((element) => {
            if (element.name === 'int8 dynamic quantization' || element.name === 'int8 static quantization') {
              const insert = Object.assign({}, element);
              this.precisionsPyTorch.push(insert);
            } else {
              const tmp = Object.assign({}, element);
              this.precisionsOther.push(tmp);
            }
          });
          if (this.data.framework === 'pytorch') {
            this.precisions = this.precisionsPyTorch;
          } else {
            this.precisions = this.precisionsOther;
          }
          this.getOptimizationTypes();
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  getOptimizationTypes() {
    this.modelService.getDictionaryWithParam('optimization_types', 'precision',
      { precision: this.precisions.find(x => x.id === this.precisionId).name })
      .subscribe(
        (response: { optimization_types: any }) => {
          this.optimizationTypes = response.optimization_types;
          const supportedTypes = this.optimizationTypes.filter(x => x.is_supported === true);
          if (supportedTypes.length > 1) {
            this.optimizationTypeId = this.optimizationTypes.find(x => x.name === 'Mixed precision').id;
          } else {
            this.optimizationTypeId = supportedTypes[0].id;
          }
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  getDatasets() {
    this.modelService.getDatasetList(this.data.projectId)
      .subscribe(
        (response: { datasets: any }) => {
          this.datasets = response.datasets;
          if (this.datasets.length > 0) {
            this.datasetId = this.datasets[0].id;
          }
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  openDatasetDialog() {
    this.modelService.openDatasetDialog$.next(true);
  }

  addOptimization() {
    if (!this.data.editing) {
      this.modelService.addOptimization({
        project_id: this.data.projectId,
        name: this.name,
        precision_id: this.precisionId,
        optimization_type_id: this.optimizationTypeId,
        dataset_id: this.datasetId
      })
        .subscribe(
          response => { this.modelService.optimizationCreated$.next(true); },
          error => {
            this.modelService.openErrorDialog(error);
          });
    } else {
      this.modelService.editOptimization({
        id: this.data.optimizationId,
        precision_id: this.precisionId,
        dataset_id: this.datasetId
      })
        .subscribe(
          response => { this.modelService.optimizationCreated$.next(true); },
          error => {
            this.modelService.openErrorDialog(error);
          });
    }
  }

}
